# Train base DINOv2
PYTHONPATH=src/ python src/train.py --config configs/train/dinov2/base.yaml

# Evaluate base DINOv2 trained with robustness tokens
#   - Evaluate robustness of features
#       - With robustness tokens
#       - Mean performance for first 6 batches: Cosine Sim: 0.922  - MSE: 0.378
PYTHONPATH=src/ python src/robustness/feat.py --config configs/robustness/features/base-rob.yaml
#       - Without robustness tokens
#       - Mean performance for first 6 batches: Cosine Sim: 0.024  - MSE: 5.183
PYTHONPATH=src/ python src/robustness/feat.py --config configs/robustness/features/base.yaml

# Train DINOv3
PYTHONPATH=src python3 src/train.py --config configs/train/dinov3/vitl16.yaml

# Evaluate DINOv3 trained with robustness tokens
#   - Evaluate robustness of features
#       - With robustness tokens
#       - Mean performance for first 6 batches: Cosine Sim: 0.864  - MSE: 0.016
PYTHONPATH=src/ python src/robustness/feat.py --config configs/robustness/features/dinov3_vitl16-rob.yaml
#       - Without robustness tokens
#       - Mean performance for first 6 batches: Cosine Sim: 0.016  - MSE: 0.167
PYTHONPATH=src/ python src/robustness/feat.py --config configs/robustness/features/dinov3_vitl16.yaml