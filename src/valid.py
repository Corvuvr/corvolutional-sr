import cv2
from datetime import datetime
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from pprint import pp
from typing import Sequence
from collections import Counter
from argparse import ArgumentParser

import data
import model
from model import weights
from utils import image, metrics, parser

# Logs
logging.basicConfig(level=logging.INFO)     # Configure logging
logger = logging.getLogger(__name__)        # Create logger for the module
args   = parser.base_parser()

class CorvolutionalLoader():
    def __init__(self, config: ArgumentParser):
        # Model
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model_config: dict = model.__dict__[config.arch](config)
        net = torch.nn.DataParallel(model_config).to(device)
        net = weights.load_checkpoint(net, device, config.checkpoint_id, strict=False)
        if config.rep:
            rep_model = weights.reparameterize(
                config=config,
                device=device,
                net=net,
                save_rep_checkpoint=False
            )
            net = rep_model
        net.eval()
        pp(model_config)
        # Datasets
        self.datasets: Sequence[dict[str, torch.utils.data.Dataset]] = list(
            dict(zip(
                ["train", "test"], 
                data.__dict__[benchmark](config).split(train_split = 0.8)
            )) 
            for benchmark in config.benchmark
        )
        self.config  = config
        self.device  = device
        self.loss_fn = torch.nn.functional.l1_loss
        self.net     = net
       
    def upscale(self, img: torch.Tensor, flo: torch.Tensor):
        if self.config.bicubic:
            out = F.interpolate(img, scale_factor=self.config.scale, mode="bicubic", align_corners=False).clamp(min=0, max=1)
        else:
            out = self.net(img, flo)
        return out
    
    @torch.no_grad()
    def evaluate(self):
        run_date = datetime.now().strftime("%Y.%m.%d-%H.%M.%S") 
        gauge = metrics.MetricGauge(log_level=logging.root.level)
        for _, dataset in enumerate(self.datasets):
            test_loader = torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                shuffle=True, 
                drop_last=False
            )
            for batch in tqdm(test_loader):
                lr_img = batch["lr"].to(self.device)
                hr_img = batch["hr"].to(self.device)
                flow   = batch["fl"].to(self.device)
                name   = batch["name"][0]
                folder = batch["folder"][0]

                # Upscale
                gauge.timer_set()
                sr_img = self.upscale(lr_img, flow)
                gauge.timer_reset()
                gauge.metrics["loss"].append(float(torch.nn.functional.l1_loss(sr_img, hr_img)))
                gauge.extract_metrics(sr_img, hr_img)

                # Save img
                SAV_FOLDER: str = "results"
                cv2_image = image.tensor2uint(sr_img * 255)
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                img_folder = f"{SAV_FOLDER}/{run_date}/{folder}"
                Path(img_folder).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{img_folder}/{name}.jpg", cv2_image)
        return gauge

    def fit(self):
        gauge = metrics.MetricGauge(log_level=logging.root.level)
        for i, dataset in enumerate(self.datasets):
            test_loader = torch.utils.data.DataLoader(
                dataset["train"],
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                shuffle=True, 
                drop_last=False
            )
            optimizer = torch.optim.AdamW(
                params=self.net.parameters(), 
                lr=1e-4,
                betas=(0.9, 0.999),
                weight_decay=0.5
            )
            for batch in tqdm(test_loader):
                with torch.no_grad():
                    lr_img = batch["lr"].to(self.device)
                    hr_img = batch["hr"].to(self.device)
                    flow   = batch["fl"].to(self.device)
                # print(f"{flow.max()=} {flow.min()=}")
                # Upscale
                gauge.timer_set()
                sr_img = self.upscale(lr_img, flow)
                gauge.timer_reset()
                
                loss = self.loss_fn(sr_img, hr_img)
                loss_val = float(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Fix memory leak with detach() 
                sr_img.detach_()
                hr_img.detach_()
                flow.detach_()
                gauge.extract_metrics(sr_img, hr_img)
                
                gauge.metrics["loss"].append(loss_val)

        torch.save(self.net.state_dict(), f"flow_model.pth")
        return gauge

if __name__ == "__main__":
    c = CorvolutionalLoader(config=args)
    num_epochs: int = 5
    num_rounds: int = 10
    train_metrics = Counter()
    test_metrics  = Counter()
    for i in range(num_rounds):
        epoch_counter = Counter()
        for epoch in range(num_epochs):
            performance = c.fit()
            epoch_counter.update(performance.avg())
        train_metrics.update(dict(map(lambda kv: (kv[0], [kv[1] / num_epochs]), dict(epoch_counter).items())))
        test_metrics .update(dict(map(lambda kv: (kv[0], [kv[1]]), c.evaluate().avg().items())))
    pp(train_metrics)
    pp(test_metrics)
        