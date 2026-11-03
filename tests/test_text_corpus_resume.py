"""Resume must preserve the actual optimizer, scheduler and crop-RNG trajectory."""

import copy

import numpy as np
import torch

from benchmarks.super_resolution.text_corpus.resume import restore


def test_checkpoint_resume_matches_uninterrupted_adam_cosine_and_crop_rng(tmp_path):
    def create():
        torch.manual_seed(682)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=6, eta_min=0.00001
        )
        return model, optimizer, scheduler, np.random.default_rng(682)

    def epoch(parts):
        model, optimizer, scheduler, rng = parts
        for _ in range(3):
            x = torch.from_numpy(rng.random((4, 3)).astype(np.float32))
            target = torch.from_numpy(rng.random((4, 2)).astype(np.float32))
            optimizer.zero_grad()
            (model(x) - target).abs().mean().backward()
            optimizer.step()
        scheduler.step()

    full = create()
    for _ in range(3):
        epoch(full)
    path = tmp_path / "last.pt"
    torch.save(
        dict(
            state_dict=copy.deepcopy(full[0].state_dict()),
            optimizer=copy.deepcopy(full[1].state_dict()),
            scheduler=full[2].state_dict(),
            numpy_rng=full[3].bit_generator.state,
            protocol_binding="fixed",
            epoch=3,
            step=9,
        ),
        path,
    )
    resumed = create()
    restore(path, *resumed, "fixed", 3)
    for _ in range(3):
        epoch(full)
        epoch(resumed)
    assert all(
        torch.equal(a, b) for a, b in zip(full[0].parameters(), resumed[0].parameters())
    )
    assert full[1].param_groups[0]["lr"] == resumed[1].param_groups[0]["lr"]
    assert np.array_equal(full[3].random(20), resumed[3].random(20))
