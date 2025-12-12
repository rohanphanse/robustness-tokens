# Evaluate classification robustness
conda activate rtokens
source .env
x=dinov2_vitl14_10_base_head
y=1
CUDA_VISIBLE_DEVICES=$y PYTHONPATH=src/ python src/robustness/classification.py \
  --config configs/robustness/classification/$x.yaml

# Evaluate feature robustness
conda activate rtokens
source .env
x=dinov2_vitl14_0
y=2
CUDA_VISIBLE_DEVICES=$y PYTHONPATH=src/ python src/robustness/feat.py --config configs/robustness/features/$x.yaml