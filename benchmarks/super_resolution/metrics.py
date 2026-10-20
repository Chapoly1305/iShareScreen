"""Paired RGB quality metrics shared by video evaluation and training."""
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def metrics(reference, output):
    if reference.shape != output.shape:
        raise ValueError(f"Geometry mismatch: {reference.shape} vs {output.shape}")
    return {
        "psnr": float(peak_signal_noise_ratio(reference, output, data_range=255)),
        "ssim": float(structural_similarity(
            reference, output, channel_axis=2, data_range=255
        )),
    }
