import numpy as np
from Qsun.Qencodes import *

ENCODING_REGISTER = {
    "YZ_CX": {
        "fn": YZ_CX_encode,
        "has_params": True,
        "has_layers": True,
    },
    "HighDim": {
        "fn": HighDim_encode,
        "has_params": False,
        "has_layers": False,
    },
    "HZY_CZ": {
        "fn": HZY_CZ_encode,
        "has_params": True,
        "has_layers": True,
    },
    "Chebyshev": {
        "fn": Chebyshev_encode,
        "has_params": True,
        "has_layers": True,
    },
    "ParamZFeatureMap": {
        "fn": ParamZFeatureMap_encode,
        "has_params": True,
        "has_layers": True,
    },
    "SeparableRX": {
        "fn": SeparableRXEncoding_encode,
        "has_params": False,
        "has_layers": False,
    },
    "HardwareEfficientRx": {
        "fn": HardwareEfficientEmbeddingRx_encode,
        "has_params": False,
        "has_layers": True,
    },
    "ZFeatureMap": {
        "fn": ZFeatureMap_encode,
        "has_params": False,
        "has_layers": True,
    },
    "ZZFeatureMap": {
        "fn": ZZFeatureMap_encode,
        "has_params": False,
        "has_layers": True,
    },
}

def encode_sample(sample: np.ndarray, encoding_name: str, n_layers: int = 2, 
                  params: np.ndarray = None):
    if encoding_name not in ENCODING_REGISTER:
        raise ValueError(f"Unknown encoding: {encoding_name}")
    
    config = ENCODING_REGISTER[encoding_name]
    fn = config["fn"]
    
    kwargs = {}
    if config["has_params"] and params is not None:
        kwargs["params"] = params
    if config["has_layers"]:
        kwargs["n_layers"] = n_layers
    
    return fn(sample, **kwargs)