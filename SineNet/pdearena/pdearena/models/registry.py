from pdearena import utils
from pdearena.modules.conditioned.twod_resnet import (
    FourierBasicBlock as CondFourierBasicBlock,
    DilatedBasicBlock as CondDilatedBasicBlock
)

MODEL_REGISTRY = {
    ## sinenet: zeros
    "sinenet1-dual-128": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 128,
            "num_waves": 1,
            "mult": 2,
            "padding_mode": "zeros",
            "par1": 0
        },
    },
    "sinenet8-dual": { # 35490219
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 8,
            "mult": 1.425,
            "padding_mode": "zeros",
            "par1": 35490219
        },
    },
    "deeper_unet8-dual": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 1,
            "num_blocks": 8,
            "mult": 1.425,
            "padding_mode": "zeros",
            "par1": 0
        },
    },
    "sinenet8-dual-tangle": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 8,
            "mult": 1.425,
            "padding_mode": "zeros",
            "disentangle": False,
            "par1": 0
        },
    },
    "sinenet6-dual": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 6,
            "mult": 1.5,
            "padding_mode": "zeros",
            "par1": 35490219
        },
    },
    "sinenet4-dual": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 4,
            "mult": 1.611,
            "padding_mode": "zeros",
            "par1": 35490219
        },
    },
    "sinenet2-dual": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 2,
            "mult": 1.8075,
            "padding_mode": "zeros",
            "par1": 35490219
        },
    },
    ## sinenet: circular
    "sinenet1-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 1,
            "mult": 2,
            "padding_mode": "circular",
            "par1": 0
        },
    },
    "sinenet8-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 8,
            "mult": 1.425,
            "padding_mode": "circular",
            "par1": 0
        },
    },
    "deeper_unet8-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 1,
            "num_blocks": 8,
            "mult": 1.425,
            "padding_mode": "circular",
            "par1": 0
        },
    },
    "sinenet8-dual-tangle-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 8,
            "mult": 1.425,
            "padding_mode": "circular",
            "disentangle": False,
            "par1": 0
        },
    },
    "sinenet6-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 6,
            "mult": 1.5,
            "padding_mode": "circular",
            "par1": 35490219
        },
    },
    "sinenet4-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 4,
            "mult": 1.611,
            "padding_mode": "circular",
            "par1": 35490219
        },
    },
    "sinenet2-dual-circ": { 
        "class_path": "pdearena.modules.sinenet_dual.sinenet",
        "init_args": {
            "hidden_channels": 64,
            "num_waves": 2,
            "mult": 1.8075,
            "padding_mode": "circular",
            "par1": 35490219
        },
    },
    # SineNet-neural-ODE
    "sinenet-neural-ODE":{
        "class_path": "pdearena.modules.sinenet_neural_ode.sinenet_node",
        "init_args": {
            "tol": 0.01,
            "hidden_channels": 64,
            "mult": 1.7,
            "padding_mode": "zeros",
        },
    }
}



