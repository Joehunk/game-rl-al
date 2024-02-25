import torch

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()  # numel gives the total number of elements, element_size gives the size in bytes of each element

    param_size_bytes = param_size  # in bytes
    param_size_megabytes = param_size_bytes / (1024 ** 2)  # Convert to megabytes

    return param_size_megabytes
