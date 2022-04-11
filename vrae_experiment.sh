# MODEL RUN
# TRAIN on diff scalers, and layers

# Standard
# python run_vrae.py --scale_type 'Standard' --hidden_layer_depth 1
# python run_vrae.py --scale_type 'Standard' --hidden_layer_depth 2

# # MinMax
python run_vrae.py --scale_type 'MinMax' --hidden_layer_depth 1 --file_name 'netis' --cols_to_remove Time label --n_epochs 1000 --is_train 'False'

# python run_vrae.py --scale_type 'MinMax' --hidden_layer_depth 2

# # Robust
# python run_vrae.py --scale_type 'Robust' --hidden_layer_depth 1
# python run_vrae.py --scale_type 'Robust' --hidden_layer_depth 2