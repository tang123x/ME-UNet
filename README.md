<div id="top"></div>

# ME-UNet

ME-UNet is a Mamba-Enhanced UNet with Conv-Mamba Hybrid Blocks for medical image segmentation.

## Setup

Check dependencies in requirements.txt, and necessarily run:

```sh
pip install -r requirements.txt
```

**Environment Requirements:**

- PyTorch 1.13.1 and Python 3.10.8

## Running

### Training

To train the model, specify the training configurations in the config file (e.g., `config/MM-WHS/ME-UNet.yaml`), then run:

```sh
python train.py --config=config/MM-WHS/ME-UNet.yaml --num_gpus 1 --num_machines 1 --machine_rank 0 --dist_url auto
```

### Testing

To test the trained model, run:

```sh
python test.py --experiment ./experiments/MM-WHS/ME-UNet --weight best.pt
```
