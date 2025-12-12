# Train robustness tokens for base DINOv2
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/ python src/train.py --config configs/train/dinov2/base_1.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/ python src/train.py --config configs/train/dinov2/base_5.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/ python src/train.py --config configs/train/dinov2/base_10.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/ python src/train.py --config configs/train/dinov2/base_20.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/ python src/train.py --config configs/train/dinov2/small_10.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src/ python src/train.py --config configs/train/dinov2/large_10.yaml

# Train robustness tokens for DINOv3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 src/train.py --config configs/train/dinov3/vits16_10.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 src/train.py --config configs/train/dinov3/vitb16_10.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 src/train.py --config configs/train/dinov3/vitl16_10.yaml

# Train linear head for DINOv2
conda activate rtokens
x=dinov2_vitl14_0
y=0
PYTHONPATH=src/ python src/convert.py \
  --checkpoint results/$x/last.ckpt \
  --output backbones/$x.pth
source .env
cd src/dinov2
CUDA_VISIBLE_DEVICES=$y PYTHONPATH=${PWD} python -m dinov2.eval.linear \
  --config-file dinov2/configs/eval/$x.yaml \
  --pretrained-weights $HOME/robustness-tokens/backbones/$x.pth \
  --output-dir $HOME/robustness-tokens/results/$x/linear \
  --train-dataset ImageNet:split=TRAIN:root=${IMAGENET_DIR}:extra=${IMAGENET_EXTRA} \
  --val-dataset ImageNet:split=VAL:root=${IMAGENET_DIR}:extra=${IMAGENET_EXTRA} \
  --epochs 3 \
  --eval-period-iterations 500

cp $HOME/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth backbones/dinov2_vits14_0.pth
cp $HOME/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth backbones/dinov2_vitb14_0.pth
cp $HOME/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth backbones/dinov2_vitl14_0.pth

conda activate rtokens
x=dinov2_vitb14_0
y=3
source .env
cd src/dinov2
CUDA_VISIBLE_DEVICES=$y PYTHONPATH=${PWD} python -m dinov2.eval.linear \
  --config-file dinov2/configs/eval/$x.yaml \
  --pretrained-weights $HOME/robustness-tokens/backbones/$x.pth \
  --output-dir $HOME/robustness-tokens/results/$x/linear \
  --train-dataset ImageNet:split=TRAIN:root=${IMAGENET_DIR}:extra=${IMAGENET_EXTRA} \
  --val-dataset ImageNet:split=VAL:root=${IMAGENET_DIR}:extra=${IMAGENET_EXTRA} \
  --epochs 3 \
  --eval-period-iterations 500

# Train linear head for DINOv3
cd $HOME/robustness-tokens/
conda activate rtokens
x=dinov3_vits16_10
PYTHONPATH=src/ python src/convert.py \
  --checkpoint results/$x/last.ckpt \
  --output backbones/$x.pth
x=dinov3_vits16_0
y=2
source .env
CUDA_VISIBLE_DEVICES=$y PYTHONPATH=dinov3/ python -m dinov3.eval.linear \
  model.config_file=$HOME/robustness-tokens/dinov3/dinov3/configs/train/$x.yaml \
  model.pretrained_weights=$HOME/robustness-tokens/backbones/$x.pth \
  output_dir=$HOME/robustness-tokens/results/$x/linear \
  train.dataset=ImageNet:split=TRAIN:root=${IMAGENET_DIR}:extra=${IMAGENET_EXTRA} \
  train.val_dataset=ImageNet:split=VAL:root=${IMAGENET_DIR}:extra=${IMAGENET_EXTRA} \
  train.checkpoint_retention_policy=LAST_AND_BEST \
  train.epochs=4 
  # train.learning_rates=[0.004] \
  # train.batch_size=128 \
  # train.optimizer_type=ADAMW

cp $HOME/.cache/torch/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth backbones/dinov3_vits16_0.pth
cp $HOME/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth backbones/dinov2_vitb14_0.pth
cp $HOME/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth backbones/dinov2_vitl14_0.pth