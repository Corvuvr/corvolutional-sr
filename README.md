# corvolutional-sr
`corvolutional-sr` is a fast [RT4KSR](https://github.com/eduardzamfir/RT4KSR)-based convolutional neural network using motion vectors for super resolution.

## Requirements
- Ubuntu 22.04 or later (or WSL)
- Support for `cuda` or `rocm`.

## Installation
```shell
conda create -n corvolutional Python=3.10
conda activate corvolutional
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
pip install -r requirements.txt  --no-input
```

## Usage
### Prepare Dataset
This model is studied based on [Sintel dataset](http://sintel.is.tue.mpg.de/downloads). Download and extract it in such a way that your project structure looks like this:
```shell
.
├── datasets
│   └── Sintel
│       ├── training
│       │   ├── clean
│       │   ├── flow
│       │   └── ...
│       └── ...
└── corvolutional-sr
```
Given this, the program should be running fine. 
Alternatively, one may inspect `src/data/benchmark.py` and create a function that returns an instance of a `Benchmark` class 
with custom `dataroot` along with paths for RGB-images and `.flo`-files containing optical flow data:
```python
def sintel_full(config):
    return Benchmark(
        dataroot=config.dataroot, # comes from --dataroot option in src/utils/parser.py
        name="Sintel",
        mode="val",
        scale=config.scale, 
        rgb_range=config.rgb_range,
        img_path="training/clean",
        flo_path="training/flow"
    )
```
Then, it is possible to use this benchmark by passing function name to the `--benchmark` option while calling `src/valid.py` script 
(see `run.sh` for example).

### Train
Training loop is specified in `src/valid.py` script. 
Run `run.sh`.