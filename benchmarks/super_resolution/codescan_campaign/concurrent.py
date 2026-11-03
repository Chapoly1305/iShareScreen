"""Shared training-batch / exclusive validation lock for one GPU."""

import fcntl
from pathlib import Path
import torch


def install(trainer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    held = False
    original_zero = torch.optim.Adam.zero_grad
    original_update = torch.amp.GradScaler.update
    original_validate = trainer.validate

    def zero(self, *args, **kwargs):
        nonlocal held
        if not held:
            fcntl.flock(handle, fcntl.LOCK_SH)
            held = True
        return original_zero(self, *args, **kwargs)

    def update(self, *args, **kwargs):
        nonlocal held
        try:
            return original_update(self, *args, **kwargs)
        finally:
            if held:
                torch.cuda.synchronize()
                fcntl.flock(handle, fcntl.LOCK_UN)
                held = False

    def validate(*args, **kwargs):
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return original_validate(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            fcntl.flock(handle, fcntl.LOCK_UN)

    torch.optim.Adam.zero_grad = zero
    torch.amp.GradScaler.update = update
    trainer.validate = validate
