"""Atomic, source-bound checkpoints for explicit continuation after pre-emption."""
import hashlib
import json
import os


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_weights(path, value):
    import torch
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save(path, model, optimizer, rng, binding, metadata, best_state):
    import torch
    payload = dict(state_dict=cpu_state(model), optimizer=optimizer.state_dict(),
                   rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                   binding=binding, metadata=metadata, best_state=best_state)
    atomic_weights(path, payload)


def load(path, model, optimizer, rng, binding):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["binding"] != binding:
        raise ValueError("Resume configuration, source or data identity changed")
    torch.nn.Module.load_state_dict(model, payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["rng"]
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["metadata"], payload["best_state"]
