"""True shuffled epochs and aligned crops, separate from the legacy sampler."""
import math


def split_indices(data, fold):
    if data.get("schema") != "sr_frozen_curated_pairs_v1" or data.get("training_ready") is not True:
        raise ValueError("A frozen curated training manifest is required")
    rows = data["pairs"]
    if any(type(r.get("validation_eligible")) is not bool
           or r["validation_eligible"] != r.get("baseline") for r in rows):
        raise ValueError("Validation must use exactly the clean baseline anchors")
    training = [i for i, r in enumerate(rows) if r["fold"] != fold]
    validation = [i for i, r in enumerate(rows) if r["fold"] == fold and r["validation_eligible"]]
    if not training or not validation:
        raise ValueError("Empty training or validation split")
    return training, validation


def epoch_order(indices, seed, epoch):
    import numpy as np
    if epoch < 0 or len(indices) != len(set(indices)):
        raise ValueError("Invalid epoch or repeated frame indices")
    # A separate generator makes the permutation reconstructible mid-epoch,
    # independent of crop RNG, model, machine and checkpoint cadence.
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0x45504F43]))
    return rng.permutation(indices).tolist()


def budget(count, batch_size, epochs):
    if min(count, batch_size, epochs) <= 0:
        raise ValueError("Positive epoch, frame and batch counts required")
    return math.ceil(count / batch_size) * epochs


def batch_at(store, order, cursor, size, rng, patch, device):
    import numpy as np
    import torch
    if not 0 <= cursor < len(order) or min(size, patch) <= 0:
        raise ValueError("Invalid batch cursor, size or patch")
    chosen = order[cursor:cursor+size]  # Keep the last partial batch.
    lows, references = [], []
    for i in chosen:
        low, reference = store.load(i)
        h, w = low.shape[:2]
        if min(h, w) < patch or reference.shape != (2*h, 2*w, 3):
            raise ValueError("Invalid aligned training geometry or patch")
        y, x = int(rng.integers(h-patch+1)), int(rng.integers(w-patch+1))
        lows.append(low[y:y+patch, x:x+patch])
        references.append(reference[2*y:2*(y+patch), 2*x:2*(x+patch)])
    def tensor(images):
        return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().to(device) / 255
    return tensor(lows), tensor(references), len(chosen)
