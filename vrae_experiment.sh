# MODEL RUN
# TRAIN on diff scalers, and layers

# Standard
python run_vrae.py --scale_type 'Standard' --hidden_layer_depth 1 --is_train False
python run_vrae.py --scale_type 'Standard' --hidden_layer_depth 2 --is_train False

# MinMax
python run_vrae.py --scale_type 'MinMax' --hidden_layer_depth 1 --is_train False
python run_vrae.py --scale_type 'MinMax' --hidden_layer_depth 2 --is_train False

# Robust
python run_vrae.py --scale_type 'Robust' --hidden_layer_depth 1 --is_train False
python run_vrae.py --scale_type 'Robust' --hidden_layer_depth 2 --is_train False