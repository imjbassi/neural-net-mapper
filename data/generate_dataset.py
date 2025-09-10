import numpy as np
from PIL import Image, ImageDraw
import random
import math


def generate_shape_image(shape, size=32, centered=True, thickness=2, jitter=2, fill=False):
    """Generate a single grayscale image with a shape.

    - centered: if True, shapes are centered with small jitter; else fully random positions
    - thickness: outline thickness for unfilled shapes
    - jitter: max pixel jitter applied to centered shapes
    - fill: fill the shape instead of outline
    """
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    def j(val):
        return val + random.randint(-jitter, jitter) if jitter > 0 else val

    if shape == "circle":
        r = random.randint(8, 12)
        if centered:
            cx, cy = size // 2, size // 2
            x0, y0 = j(cx - r), j(cy - r)
            x1, y1 = x0 + 2 * r, y0 + 2 * r
        else:
            x0 = random.randint(0, size - r * 2)
            y0 = random.randint(0, size - r * 2)
            x1, y1 = x0 + 2 * r, y0 + 2 * r
        if fill:
            draw.ellipse([x0, y0, x1, y1], fill=0)
        else:
            draw.ellipse([x0, y0, x1, y1], outline=0, width=thickness)

    elif shape == "square":
        s = random.randint(12, 16)
        if centered:
            cx, cy = size // 2, size // 2
            x0, y0 = j(cx - s // 2), j(cy - s // 2)
            x1, y1 = x0 + s, y0 + s
        else:
            x0 = random.randint(0, size - s)
            y0 = random.randint(0, size - s)
            x1, y1 = x0 + s, y0 + s
        if fill:
            draw.rectangle([x0, y0, x1, y1], fill=0)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=0, width=thickness)

    elif shape == "triangle":
        # Use an approximately equilateral triangle centered
        side = random.randint(14, 18)
        h = side * math.sqrt(3) / 2
        if centered:
            cx, cy = size // 2, size // 2
            pts = [
                (j(int(cx)), j(int(cy - 2*h/3))),  # top
                (j(int(cx - side/2)), j(int(cy + h/3))),  # bottom-left
                (j(int(cx + side/2)), j(int(cy + h/3))),  # bottom-right
            ]
        else:
            # Random triangle but non-degenerate
            pts = [
                (random.randint(4, size-4), random.randint(4, size-4)),
                (random.randint(4, size-4), random.randint(4, size-4)),
                (random.randint(4, size-4), random.randint(4, size-4)),
            ]
        if fill:
            draw.polygon(pts, fill=0)
        else:
            draw.polygon(pts, outline=0)

    return np.array(img) / 255.0


def generate_dataset(n_per_class=100, size=32, centered=True, thickness=2, jitter=2, fill=False):
    shapes = ["circle", "square", "triangle"]
    X = []
    y = []
    for idx, shape in enumerate(shapes):
        for _ in range(n_per_class):
            img = generate_shape_image(shape, size=size, centered=centered, thickness=thickness, jitter=jitter, fill=fill)
            X.append(img.flatten())
            y.append(idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y
