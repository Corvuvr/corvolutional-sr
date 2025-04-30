import logging
import torch
import torch.nn.functional as F
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
        gauge = metrics.MetricGauge(log_level=logging.root.level)
        for i, dataset in enumerate(self.datasets):
            test_loader = torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=1,
                num_workers=1,
                pin_memory=False,
                shuffle=False, 
                drop_last=False
            )
            for batch in tqdm(test_loader):
                lr_img = batch["lr"].to(self.device)
                hr_img = batch["hr"].to(self.device)
                flow   = batch["fl"].to(self.device)
                # Upscale
                gauge.timer_set()
                sr_img = self.upscale(lr_img, flow)
                gauge.timer_reset()
                
                gauge.extract_metrics(sr_img, hr_img)
        return gauge

    def fit(self):
        gauge = metrics.MetricGauge(log_level=logging.root.level)
        for i, dataset in enumerate(self.datasets):
            test_loader = torch.utils.data.DataLoader(
                dataset["train"],
                batch_size=1,
                num_workers=1,
                pin_memory=False,
                shuffle=False, 
                drop_last=False
            )
            optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-4)  
            for batch in tqdm(test_loader):
                with torch.no_grad():
                    lr_img = batch["lr"].to(self.device)
                    hr_img = batch["hr"].to(self.device)
                    flow   = batch["fl"].to(self.device)
                # Upscale
                gauge.timer_set()
                sr_img = self.upscale(lr_img, flow)
                gauge.timer_reset()
                
                loss = self.loss_fn(sr_img, hr_img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Fix memory leak with detach() 
                gauge.extract_metrics(sr_img.detach(), hr_img.detach())
        return gauge

if __name__ == "__main__":
    c = CorvolutionalLoader(config=args)
    num_epochs: int = 1
    num_rounds: int = 2
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
        