"""2D affine transform composition with matrix chaining.

Convention: row-major 3x3 matrices, points as column vectors
multiplied on the right (M @ p). Composition is left-multiplication
so chained calls apply right-to-left: translate(a).rotate(b) means
first rotate, then translate.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


_EPS = 1e-12


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """3x3 matrix product a @ b."""
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]


def _identity() -> List[List[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _valid(m: Sequence[Sequence[float]]) -> bool:
    return len(m) == 3 and all(len(row) == 3 for row in m)


class AffineTransform:
    """A 2D affine transform held as a row-major 3x3 matrix.

    Operations return new instances so chains compose without surprises.
    The bottom row is always [0, 0, 1]; non-affine matrices cannot be
    constructed or produced, which is what callers expect from a 2D
    affine library.
    """

    __slots__ = ("_m",)

    def __init__(self, matrix: Sequence[Sequence[float]] | None = None) -> None:
        if matrix is None:
            self._m = _identity()
        else:
            if not _valid(matrix):
                raise ValueError("matrix must be 3x3")
            # Always coerce to a pure float list so downstream callers
            # never trip on ints mixed with floats.
            self._m = [[float(v) for v in row] for row in matrix]

    @staticmethod
    def identity() -> "AffineTransform":
        """Return an identity transform."""
        return AffineTransform()

    @staticmethod
    def translation(dx: float, dy: float) -> "AffineTransform":
        """Return a pure translation transform."""
        return AffineTransform([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]])

    @staticmethod
    def rotation(angle_radians: float) -> "AffineTransform":
        """Return a rotation about the origin (radians)."""
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        return AffineTransform([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def scaling(sx: float, sy: float) -> "AffineTransform":
        """Return a non-uniform scale about the origin."""
        return AffineTransform([[float(sx), 0.0, 0.0], [0.0, float(sy), 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def shearing(kx: float, ky: float) -> "AffineTransform":
        """Return a shear about the origin."""
        return AffineTransform([[1.0, float(kx), 0.0], [float(ky), 1.0, 0.0], [0.0, 0.0, 1.0]])

    def _chain(self, other: "AffineTransform") -> "AffineTransform":
        return AffineTransform(_matmul(self._m, other._m))

    def translate(self, dx: float, dy: float) -> "AffineTransform":
        """Append a translation to the chain (returns new transform)."""
        return self._chain(AffineTransform.translation(dx, dy))

    def rotate(self, angle_radians: float) -> "AffineTransform":
        """Append a rotation (radians) to the chain (returns new transform)."""
        return self._chain(AffineTransform.rotation(angle_radians))

    def rotate_degrees(self, angle_degrees: float) -> "AffineTransform":
        """Append a rotation given in degrees (returns new transform)."""
        return self.rotate(math.radians(angle_degrees))

    def scale(self, sx: float, sy: float) -> "AffineTransform":
        """Append a scale to the chain (returns new transform)."""
        return self._chain(AffineTransform.scaling(sx, sy))

    def shear(self, kx: float, ky: float) -> "AffineTransform":
        """Append a shear to the chain (returns new transform)."""
        return self._chain(AffineTransform.shearing(kx, ky))

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        """Apply this transform to a single point (x, y)."""
        m = self._m
        nx = m[0][0] * x + m[0][1] * y + m[0][2]
        ny = m[1][0] * x + m[1][1] * y + m[1][2]
        return (nx, ny)

    def matrix(self) -> List[List[float]]:
        """Return a row-major copy of the 3x3 matrix as floats."""
        return [row[:] for row in self._m]

    def __repr__(self) -> str:
        return f"AffineTransform({self._m!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineTransform):
            return NotImplemented
        a, b = self._m, other._m
        for i in range(3):
            for j in range(3):
                if abs(a[i][j] - b[i][j]) > _EPS:
                    return False
        return True

    def __hash__(self) -> int:
        # Hash on rounded entries so near-equal transforms hash equal.
        return hash(tuple(round(v, 9) for row in self._m for v in row))
