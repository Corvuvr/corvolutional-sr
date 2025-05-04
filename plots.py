import json

from pprint import pp
from typing import Sequence
from pathlib import Path
from functools import cmp_to_key

import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from PIL import Image

class Record():
    def __init__(self, filepath):
        with open(filepath) as file:
            f = json.load(file)
            self.config:        dict = f[0]
            self.train_metrics: dict = f[1]
            self.test_metrics:  dict = f[2]
        self.loss_type: str = self.config['loss']
        self.fwd_type:  str = self.config['forward_option']

def cmp_by_loss(i1: Record, i2: Record):
    return 1 if (i1.loss_type > i2.loss_type) else -1

def cmp_by_fwd(i1: Record, i2: Record):
    return 1 if (i1.loss_type > i2.loss_type) else -1

def plot_to_numpy(record: Record, metric: str, figsize=(4, 3), dpi=100) -> np.ndarray:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    match record.fwd_type:
        case 'vanilla_hf':
            fwd_type = 'vanilla'            
        case 'flow':
            fwd_type = 'flow (sep)'
        case 'flow_cat':
            fwd_type = 'flow (solid)'

    ax.set_title(f'Arch: {fwd_type}. Loss: {record.loss_type}.')
    ax.plot(record.test_metrics[metric])
    ax.plot(record.train_metrics[metric])
    ylim: tuple[float] = ...
    match metric:
        case 'psnr':
            ylim = (20, 33.25)
        case 'ssim':
            ylim = (0.65, 0.935)
        case 'time':
            ylim = (0.005, 0.012)
        
    ax.set_ylim(ylim)
    # ax.axis('off')  # rm axis
    canvas = FigureCanvas(fig)
    
    buf = BytesIO()
    canvas.print_png(buf)
    buf.seek(0)

    image = Image.open(buf).convert('RGB')
    array = np.array(image)

    plt.close(fig)
    return array

def sav_plt(records: Sequence, metric: str, filepath: str = 'plots'):
    Path(filepath).mkdir(parents=True, exist_ok=True)
    images = [
        [plot_to_numpy(r, metric=metric) for r in bank] 
        for bank in records
    ]
    plot = np.concatenate([np.concatenate(row, axis=1) for row in images], axis=0)
    cv2.imwrite(filename=f'{filepath}/{metric}.jpg', img=plot)


json_files: Sequence[str | Path] = [
    # ========= PASTE YOUR METRICS HERE =========
    # Use 'find . -name metrics.json' command in bash
    "./results/2025.05.03-07.18.04/metrics.json",
    "./results/2025.05.03-09.17.17/metrics.json",
    "./results/2025.05.03-11.04.44/metrics.json",
    "./results/2025.05.03-12.51.37/metrics.json",
    "./results/2025.05.03-14.38.19/metrics.json",
    "./results/2025.05.03-16.20.38/metrics.json",
    "./results/2025.05.03-18.07.01/metrics.json",
    "./results/2025.05.03-19.55.44/metrics.json",
    "./results/2025.05.03-21.40.32/metrics.json",
]
records: Sequence[Record] = [Record(filepath) for filepath in json_files]
records = sorted(records, key=cmp_to_key(cmp_by_fwd))
records = np.split(np.array(records), 3)
for bank in records:
    bank = sorted(bank, key=cmp_to_key(cmp_by_loss))

sav_plt(records, metric='psnr')
sav_plt(records, metric='ssim')

for bank in records:
    pp([ max(r.test_metrics['ssim']) for r in bank])
    # pp([[r.loss_type, r.fwd_type] for r in bank])
    # pp([np.argmax(r.test_metrics['psnr']) for r in bank])
