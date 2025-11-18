# Train base DINOv2
PYTHONPATH=src/ python src/train.py --config configs/train/dinov2/base.yaml

# Evaluate base DINOv2 trained with robustness tokens
#   - Evaluate robustness of features
PYTHONPATH=src/ python src/robustness/feat.py --config configs/robustness/features/base-rob.yaml
#   - Evaluate downstream classification performance

# Train DINOv3
PYTHONPATH=src python3 src/train.py --config configs/train/dinov3/vitl16.yaml
