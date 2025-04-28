import time
import torch
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict

import data
import model
from model import weights
from utils import image, metrics, parser

args = parser.base_parser()

def valid(config):

    # Metrics
    test_results = OrderedDict()
    test_results["psnr_rgb"] = []
    test_results["psnr_y"] = []
    test_results["ssim_rgb"] = []
    test_results["ssim_y"] = []
    test_results["time"] = []
    test_results = test_results

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

    # Validation
    for benchmark in config.benchmark:
        test_loader = torch.utils.data.DataLoader(
            data.__dict__[benchmark](config),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False, 
            drop_last=False
        )
        with torch.no_grad():
            print("Testing...")   
            for batch in tqdm(test_loader):
                lr_img = batch["lr"].to(device)
                hr_img = batch["hr"].to(device)
                
                # run method
                start = time.time()
                if config.bicubic:
                    out = F.interpolate(lr_img, scale_factor=config.scale, mode="bicubic", align_corners=False).clamp(min=0, max=1)
                else:
                    out = net(lr_img)
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

            print(f"------> Results of X{config.scale} for benchmark: {benchmark}")
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
    valid(args)