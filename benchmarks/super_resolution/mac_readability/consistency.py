"""Image-only regularized luminance backprojection; experimental, not deployed."""

import numpy as np
from scipy.sparse import csr_matrix


def lanczos_matrix(source, target):
    scale = source / target
    assert scale >= 1
    rows = []
    columns = []
    values = []
    for dest in range(target):
        center = (dest + 0.5) * scale - 0.5
        indexes = np.arange(
            int(np.ceil(center - 3 * scale)), int(np.floor(center + 3 * scale)) + 1
        )
        distance = (indexes - center) / scale
        weights = np.sinc(distance) * np.sinc(distance / 3)
        weights[np.abs(distance) >= 3] = 0
        weights /= weights.sum()
        rows.extend([dest] * len(indexes))
        columns.extend(np.clip(indexes, 0, source - 1))
        values.extend(weights)
    return csr_matrix(
        (np.asarray(values, np.float32), (rows, columns)), shape=(target, source)
    )


class Projection:
    def __init__(self, height=1080, width=1920):
        self.vertical = lanczos_matrix(height, height * 2 // 3)
        self.horizontal = lanczos_matrix(width, width * 2 // 3)

    def down(self, image):
        return self.horizontal.dot(self.vertical.dot(image).T).T

    def adjoint(self, image):
        return self.vertical.T.dot(self.horizontal.T.dot(image.T).T)


def luminance(rgb):
    return rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def correct(prior, low, operator, iterations):
    reference = luminance(prior)
    target = luminance(low)
    current = reference.copy()
    result = {}
    original = prior.astype(np.float32)
    for step in range(1, max(iterations) + 1):
        error = target - operator.down(current)
        error = np.sign(error) * np.maximum(np.abs(error) - 1.0, 0.0)
        delta = 1.5 * (operator.adjoint(error) - 0.1 * (current - reference))
        correction = np.clip(current + delta - reference, -16, 16)
        output = np.clip(original + correction[:, :, None], 0, 255)
        current = luminance(output)
        if step in iterations:
            result[step] = np.rint(output).astype(np.uint8)
    return result
