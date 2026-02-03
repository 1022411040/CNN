import torch
import torch.nn as nn
from datetime import datetime

class NeuronMonitor:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.activations = {}

    def _hook(self, name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                self.activations[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "max": out.max().item(),
                    "min": out.min().item(),
                    "shape": list(out.shape),
                    "timestamp": datetime.utcnow().isoformat()
                }
        return fn

    def attach(self):
        self.detach()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                h = module.register_forward_hook(self._hook(name))
                self.handles.append(h)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def snapshot(self):
        return self.activations
