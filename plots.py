import os
import json
import subprocess
import datetime as dt
from typing import Sequence
from pathlib import Path
from pprint import pp
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


# THIS CODE ASSUMES YOU HAVE ALREADY RUN TESTS WITH:
# 3 loss functions - ['l1_msssim', 'l1', 'l2'] 
# 3 architectures  - ['vanilla_hf', 'flow', 'flow_cat'] 
# 3 * 3 = 9 JSON FILES IN TOTAL 
# PASTE MODIFIED DATE OF THE FIRST OF 9 TEST FILES BELOW:
since: str = '2025.05.03-07.18.04'
up_to: int = 9

cmd: str = 'find . -name metrics.json'
cmd_output: str = subprocess.run(cmd.split(sep=' '), capture_output=True).stdout.decode()
json_files: Sequence[str | Path] = []
for file in cmd_output.split():
    date_modified: float = os.path.getmtime(file)
    since_float:   float = dt.datetime.strptime(since, "%Y.%m.%d-%H.%M.%S").timestamp()
    if date_modified >= since_float:
        json_files.append(file)
        if len(json_files) >= up_to:
            break

records: Sequence[Record] = [Record(filepath) for filepath in json_files]
records = sorted(records, key=cmp_to_key(cmp_by_fwd))
records = np.split(np.array(records), 3)
for bank in records:
    bank = sorted(bank, key=cmp_to_key(cmp_by_loss))

sav_plt(records, metric='psnr')
sav_plt(records, metric='ssim')

print("\nMAX SSIM ACROSS CLIENTS:")
for bank in records:
    pp([ max(r.test_metrics['ssim']) for r in bank])
    # pp([[r.loss_type, r.fwd_type] for r in bank])
    # pp([np.argmax(r.test_metrics['psnr']) for r in bank])
