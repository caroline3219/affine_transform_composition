# affine_transform_composition

Compose 2D affine transformations (translation, rotation, scale, shear) into a single 3×3 matrix. Each operation is a method returning a new transform so chains stay readable.

```python
from affine_transform_composition import AffineTransform

t = (AffineTransform()
     .translate(100, 50)
     .scale(2, 2)
     .rotate(3.141592653589793))

x, y = t.apply(5, 3)
mat = t.matrix()  # row-major list of 3 lists, length 3
```

## Why

Most affine libraries conflate construction and application. Building a transform one step at a time is the common need, and pushing the matrix into the caller's hands every step is noise. This library gives each primitive a chaining method and holds the matrix internally.

A single trade-off shapes the API: no inverse is provided. Matrices are kept in row-major float form with no decomposition round-trip; composing more than two transforms inverts the reading order once and does not do anything else. If an inverse is needed, multiply the 3×3 matrix downstream.

## Edge

Composed transforms apply right-to-left. `translate(100, 0).rotate(θ).apply(0, 0)` means: first rotate the point at the origin, then translate it. This matches matrix multiplication order (`T · R · p`) but trips people who read chains left-to-right.

## Exported names

- `AffineTransform` — the class.
- `AffineTransform.identity()` — returns an identity transform.
- `AffineTransform(matrix)` — construct from a row-major 3×3 list.
- `.translate(dx, dy)`, `.rotate(angle_radians)`, `.rotate_degrees(angle_degrees)`, `.scale(sx, sy)`, `.shear(kx, ky)` — return new `AffineTransform`.
- `.apply(x, y)` — return `(x', y')` tuple of floats.
- `.matrix()` — return row-major list of 3 lists, length 3, floats.
