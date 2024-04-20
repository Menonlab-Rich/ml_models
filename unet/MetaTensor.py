import torch

class MetaTensor:
    def __init__(self, tensor, **metadata):
        self._tensor = tensor
        self._metadata = metadata
    def __getattr__(self, name):
        # This method is called only if there isn't an attribute in TensorProxy
        attr = getattr(self._tensor, name)
        if callable(attr):
            def _hook(*args, **kwargs):
                # This allows the class to still use the tensor's methods
                result = attr(*args, **kwargs)
                if isinstance(result, torch.Tensor):
                    return MetaTensor(result, **self._metadata)  # Ensure the tag stays with tensor operations that return tensors
                return result
            return _hook
        return attr

    def __getitem__(self, idx):
        # This is necessary for indexing and slicing operations
        result = self._tensor[idx]
        return MetaTensor(result, **self._metadata) if isinstance(result, torch.Tensor) else result

    def to(self, device):
        # Custom method to handle device transfer while maintaining tag
        return MetaTensor(self._tensor.to(device), **self._metadata)

    def __repr__(self):
        return f"{self._tensor.__repr__()} with metadata {self._metadata}"
