# AttentionPainter


## Prepare

Our code is trained on NVIDIA RTX 4090 GPU with 24GB

```bash
conda create -n attentionpainter python=3.8.5
conda activate attentionpainter
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip3 install timm==0.6.12 opencv-python==4.1.2.30 numpy==1.20.3 pillow==9.5.0 tqdm==4.65.0 tensorboard==2.13.0 six==1.17.0
```

## Training Step

### (0) Prepare
Checkpoints prepare: Download the [Neural Renderer](https://drive.google.com/file/d/1meZL9ayCKZGYYrFbisOI4wNonuYnTtEV/view) checkpoint.

Data prepare: Prepare the ImageNet Dataset.

### (1) Train AttentionPainter

run:

```
python3 main_pretrain_oil_density_w_FSS.py --data_path=$PATH_TO_IMAGENET --nr_path=$PATH_TO_NEURAL_RENDERER
```

### (2) Test AttentionPainter

run:

```
python3 test_batch_oil_density_v2.py --img_dir=$PATH_TO_TEST_IMAGES --ckpt=$PATH_TO_TRAINED_CHECKPOINT --row_divide=$ROW_NUMBER --col_divide=$COLUMN_NUMBER --output_dir=$PATH_TO_OUTPUT
```
