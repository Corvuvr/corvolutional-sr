import time
import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Sequence
from argparse import ArgumentParser
from collections import OrderedDict

import data
import model
from model import weights
from utils import image, metrics, parser

args = parser.base_parser()

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
    
    def evaluate(self):
        # Metrics
        test_results = OrderedDict()
        test_results["psnr_rgb"] = []
        test_results["psnr_y"] = []
        test_results["ssim_rgb"] = []
        test_results["ssim_y"] = []
        test_results["time"] = []
        test_results = test_results

        for i, dataset in enumerate(self.datasets):
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                shuffle=False, 
                drop_last=False
            )
            with torch.no_grad():
                print("Evaluating...")   
                for batch in tqdm(test_loader):
                    lr_img = batch["lr"].to(self.device)
                    hr_img = batch["hr"].to(self.device)
                    
                    # run method
                    start = time.time()
                    if self.config.bicubic:
                        out = F.interpolate(lr_img, scale_factor=self.config.scale, mode="bicubic", align_corners=False).clamp(min=0, max=1)
                    else:
                        out = self.net(lr_img)
                    end = time.time() - start
                    out *= 255.
                    out = image.tensor2uint(out)
                    hr_img *= 255.
                    hr_img = image.tensor2uint(hr_img)
                    
                    test_results["time"]    .append(end)
                    test_results["psnr_rgb"].append(metrics.calculate_psnr(out, hr_img, crop_border=0))
                    test_results["ssim_rgb"].append(metrics.calculate_ssim(out, hr_img, crop_border=0))
                    test_results["psnr_y"]  .append(metrics.calculate_psnr(out, hr_img, crop_border=0, test_y_channel=True))
                    test_results["ssim_y"]  .append(metrics.calculate_ssim(out, hr_img, crop_border=0, test_y_channel=True))

                print(f"------> Results of X{self.config.scale} for benchmark: {self.config.benchmark[i]}")
                ave_psnr_rgb = sum(test_results["psnr_rgb"]) / len(test_results["psnr_rgb"])
                print('------> Average PSNR (RGB): {:.6f} dB'.format(ave_psnr_rgb))
                ave_ssim_rgb = sum(test_results["ssim_rgb"]) / len(test_results["ssim_rgb"])
                print('------> Average SSIM (RGB): {:.6f}'.format(ave_ssim_rgb))
                ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
                print('------> Average PSNR (Y): {:.6f} dB'.format(ave_psnr_y))
                ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"]) 
                print('------> Average SSIM (Y): {:.6f}'.format(ave_ssim_y))
                ave_time = sum(test_results["time"]) / len(test_results["time"]) 
                print('------> Average Time: {:.6f}'.format(ave_time))     

if __name__ == "__main__":
    c = CorvolutionalLoader(config=args)
    c.evaluate()