# corvolutional-sr
`corvolutional-sr` is a fast [RT4KSR](https://github.com/eduardzamfir/RT4KSR)-based convolutional neural network using motion vectors for super resolution.

## Installation
```shell
conda create -n corvolutional Python=3.10
conda activate corvolutional
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia
pip install -r requirements.txt  --no-input
```

## Use
Run `infer.sh`.