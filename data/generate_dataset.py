```python
import math
import random

import numpy as np
from PIL import Image, ImageDraw


def generate_shape_image(
    shape, size=32, centered=True, thickness=2, jitter=2, fill=False
):
    """Generate a single grayscale image with a shape.

    Args:
        shape: Type of shape to generate ('circle', 'square', or 'triangle')
        size: Image dimensions (size x size pixels)
        centered: If True, shapes are centered with small jitter; else fully random positions
        thickness: Outline thickness for unfilled shapes
        jitter: Maximum pixel jitter applied to centered shapes
        fill: Fill the shape instead of drawing outline only

    Returns:
        numpy.ndarray: Normalized grayscale image array with values in [0, 1]
    """
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    def apply_jitter(val):
        """Apply random jitter to a coordinate value."""
        return val + random.randint(-jitter, jitter) if jitter > 0 else val

    if shape == "circle":
        radius = random.randint(8, 12)
        if centered:
            center_x, center_y = size // 2, size // 2
            x0 = apply_jitter(center_x - radius)
            y0 = apply_jitter(center_y - radius)
        else:
            x0 = random.randint(0, max(0, size - 2 * radius))
            y0 = random.randint(0, max(0, size - 2 * radius))

        x1 = x0 + 2 * radius
        y1 = y0 + 2 * radius

        if fill:
            draw.ellipse([x0, y0, x1, y1], fill=0)
        else:
            draw.ellipse([x0, y0, x1, y1], outline=0, width=thickness)

    elif shape == "square":
        side_length = random.randint(12, 16)
        if centered:
            center_x, center_y = size // 2, size // 2
            x0 = apply_jitter(center_x - side_length // 2)
            y0 = apply_jitter(center_y - side_length // 2)
        else:
            x0 = random.randint(0, max(0, size - side_length))
            y0 = random.randint(0, max(0, size - side_length))

        x1 = x0 + side_length
        y1 = y0 + side_length

        if fill:
            draw.rectangle([x0, y0, x1, y1], fill=0)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=0, width=thickness)

    elif shape == "triangle":
        side_length = random.randint(14, 18)
        height = side_length * math.sqrt(3) / 2

        if centered:
            center_x, center_y = size // 2, size // 2
            # Equilateral triangle with centroid at (center_x, center_y)
            points = [
                (
                    apply_jitter(int(center_x)),
                    apply_jitter(int(center_y - 2 * height / 3)),
                ),
                (
                    apply_jitter(int(center_x - side_length / 2)),
                    apply_jitter(int(center_y + height / 3)),
                ),
                (
                    apply_jitter(int(center_x + side_length / 2)),
                    apply_jitter(int(center_y + height / 3)),
                ),
            ]
        else:
            # Random triangle with vertices within image bounds
            points = [
                (random.randint(4, size - 4), random.randint(4, size - 4)),
                (random.randint(4, size - 4), random.randint(4, size - 4)),
                (random.randint(4, size - 4), random.randint(4, size - 4)),
            ]

        if fill:
            draw.polygon(points, fill=0)
        else:
            draw.polygon(points, outline=0, width=thickness)

    else:
        raise ValueError(f"Unknown shape: {shape}. Must be 'circle', 'square', or 'triangle'")

    return np.array(img, dtype=np.float32) / 255.0


def generate_dataset(
    n_per_class=100, size=32, centered=True, thickness=2, jitter=2, fill=False
):
    """Generate a dataset of shape images with labels.

    Args:
        n_per_class: Number of samples to generate per shape class
        size: Image dimensions (size x size pixels)
        centered: If True, shapes are centered with small jitter; else fully random positions
        thickness: Outline thickness for unfilled shapes
        jitter: Maximum pixel jitter applied to centered shapes
        fill: Fill the shapes instead of drawing outline only

    Returns:
        tuple: (X, y) where X is array of flattened images and y is array of labels
            - X shape: (n_samples, size*size)
            - y shape: (n_samples,)
            - Labels: 0=circle, 1=square, 2=triangle
    """
    shapes = ["circle", "square", "triangle"]
    X = []
    y = []

    for label, shape in enumerate(shapes):
        for _ in range(n_per_class):
            img = generate_shape_image(
                shape,
                size=size,
                centered=centered,
                thickness=thickness,
                jitter=jitter,
                fill=fill,
            )
            X.append(img.flatten())
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y
```