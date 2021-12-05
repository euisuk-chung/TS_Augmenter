# Window 30, batch 1, undo False Fixed
# Robust, layer1
python ./main.py\
    --hidden_layer_depth=1\
    --scale_type='Robust'
Robust
# MINMAX, layer2
python ./main.py\
    --hidden_layer_depth=2\
    --scale_type='Robust'

# MINMAX, layer1
python ./main.py\
    --hidden_layer_depth=1\
    --scale_type='MinMax'
    
# MINMAX, layer2
python ./main.py\
    --hidden_layer_depth=2\
    --scale_type='MinMax'

# Standard, layer1
python ./main.py\
    --hidden_layer_depth=1\
    --scale_type='Standard'
    
# Standard, layer2
python ./main.py\
    --hidden_layer_depth=2\
    --scale_type='Standard'
