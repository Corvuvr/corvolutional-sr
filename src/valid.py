import logging
import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Sequence
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
        net = torch.nn.DataParallel(model.__dict__[config.arch](config)).to(device)
        net = weights.load_checkpoint(net, device, config.checkpoint_id)
        if config.rep:
            rep_model = weights.reparameterize(
                config=config,
                device=device,
                net=net,
                save_rep_checkpoint=False
            )
            net = rep_model
        net.eval()
        # Datasets
        self.datasets: Sequence[torch.utils.data.Dataset] = [data.__dict__[benchmark](config) for benchmark in config.benchmark]
        self.config = config
        self.device = device
        self.net = net
    
    def upscale(self, img: torch.Tensor):
        if self.config.bicubic:
            out = F.interpolate(img, scale_factor=self.config.scale, mode="bicubic", align_corners=False).clamp(min=0, max=1)
        else:
            out = self.net(img)
        return out
    
    @torch.no_grad()
    def evaluate(self):
        gauge = metrics.MetricGauge(log_level=logging.root.level)
        for i, dataset in enumerate(self.datasets):
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                shuffle=False, 
                drop_last=False
            )
            logger.info("Evaluating...")   
            for batch in tqdm(test_loader):
                lr_img = batch["lr"].to(self.device)
                hr_img = batch["hr"].to(self.device)
                # Upscale
                gauge.timer_set()
                sr_img = self.upscale(lr_img)
                gauge.timer_reset()
                sr_img *= 255.
                sr_img = image.tensor2uint(sr_img)
                hr_img *= 255.
                hr_img = image.tensor2uint(hr_img)
                gauge.extract_metrics(sr_img, hr_img)
        gauge.summary()

if __name__ == "__main__":
    c = CorvolutionalLoader(config=args)
    c.evaluate()