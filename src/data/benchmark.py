import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List

from PIL import Image
import matplotlib.pyplot as plt

from data import transforms
from data.basedataset import BaseDataset


class Benchmark(BaseDataset):
    def __init__(self,
                 dataroot: str,
                 name: str,
                 mode: str,
                 img_path: str,
                 flo_path: str,
                 scale: int,
                 crop_size: int = 64,
                 rgb_range: int = 1
                 ) -> None:
        super(Benchmark, self).__init__(dataroot=dataroot,
                                        name=name,
                                        mode=mode,
                                        scale=scale,
                                        crop_size=crop_size,
                                        rgb_range=rgb_range)

        self.hr_dir_path = Path(dataroot).resolve().joinpath(img_path).__str__()
        self.fl_dir_path = Path(dataroot).resolve().joinpath(flo_path).__str__()

        self.fl_files = [os.path.join(self.fl_dir_path, x) for x in sorted(os.listdir(self.fl_dir_path))]
        self.hr_files = [os.path.join(self.hr_dir_path, x) for x in sorted(os.listdir(self.hr_dir_path))]

        print(f"{len(self.fl_files)=}")
        self.transforms = transforms.Compose([
            transforms.ToTensor(rgb_range=self.rgb_range)
        ])
        self.degrade = transforms.BicubicDownsample(scale)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_index(index)
        hr = Image.open(self.hr_files[idx]).convert("RGB")
        hr = self.transforms(hr)
        lr, hr = self.degrade(hr)
        try:
            fl: torch.Tensor = read_flo_file(self.fl_files[idx])
        except IndexError:
            fl: torch.Tensor = torch.empty(size=hr.shape)

        # assert that images are divisable by 2
        c, lr_h, lr_w = lr.shape
        lr_hr, lr_wr = int(lr_h/2), int(lr_w/2)
        lr = lr[:, :lr_hr*2, :lr_wr*2]
        hr = hr[:, :lr.shape[-2] * self.scale, :lr.shape[-1] * self.scale]
        assert lr.shape[-1] * self.scale == hr.shape[-1]
        assert lr.shape[-2] * self.scale == hr.shape[-2]

        return {
            "lr": lr.to(torch.float32), 
            "hr": hr.to(torch.float32),
            "fl": fl.to(torch.float32),
        }
    
    def __len__(self):
        return len(self.hr_files)
       
def read_flo_file(filename: str) -> torch.Tensor:
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError('Invalid .flo file')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        data = np.resize(data, (h, w, 2))  # (H, W, 2)
        return torch.from_numpy(data)  # Возвращаем в формате PyTorch Tensor

def flow_to_image(flow):
    """
    Принимает тензор (H, W, 2) и возвращает изображение в формате uint8 (H, W, 3)
    """
    flow = flow.numpy()  # в numpy
    u = flow[..., 0]
    v = flow[..., 1]

    # Ограничение максимальных значений для стабильности
    rad = np.sqrt(u**2 + v**2)
    maxrad = np.max(rad)
    epsilon = 1e-5
    u = u / (maxrad + epsilon)
    v = v / (maxrad + epsilon)

    # Переводим в формат HSV
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = (np.arctan2(-v, -u) / np.pi + 1) / 2  # hue (от 0 до 1)
    hsv[..., 1] = 1.0  # saturation
    hsv[..., 2] = np.clip(rad / (maxrad + epsilon), 0, 1)  # value (яркость)
    hsv = (hsv * 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb

def show_flow(flow):
    img = flow_to_image(flow)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Optical Flow Visualization')
    plt.show()

def sintel(config):
    return Benchmark(
        dataroot=config.dataroot, 
        name="Sintel",
        mode="val",
        scale=config.scale, 
        rgb_range=config.rgb_range,
        img_path="training/clean/mountain_1",
        flo_path="training/flow/mountain_1"
    )
