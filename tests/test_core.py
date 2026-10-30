import math
import unittest

from affine_transform_composition import AffineTransform


def approx(a, b, eps=1e-9):
    return abs(a - b) < eps


def mat_approx(a, b, eps=1e-9):
    for i in range(3):
        for j in range(3):
            if abs(a[i][j] - b[i][j]) > eps:
                return False
    return True


class TestConstruction(unittest.TestCase):
    def test_default_is_identity(self):
        t = AffineTransform()
        self.assertTrue(mat_approx(t.matrix(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_identity_static(self):
        t = AffineTransform.identity()
        self.assertTrue(mat_approx(t.matrix(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_construct_from_int_matrix(self):
        # ints are coerced to floats; this is the documented contract.
        t = AffineTransform([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
        m = t.matrix()
        self.assertIsInstance(m[0][2], float)
        self.assertTrue(approx(m[0][2], 3.0))

    def test_construct_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            AffineTransform([[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            AffineTransform([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])


class TestApply(unittest.TestCase):
    def test_identity_unchanged(self):
        t = AffineTransform()
        x, y = t.apply(5.0, -3.0)
        self.assertTrue(approx(x, 5.0))
        self.assertTrue(approx(y, -3.0))

    def test_translate(self):
        t = AffineTransform.translation(10.0, -5.0)
        x, y = t.apply(1.0, 1.0)
        self.assertTrue(approx(x, 11.0))
        self.assertTrue(approx(y, -4.0))

    def test_scale(self):
        t = AffineTransform.scaling(2.0, 0.5)
        x, y = t.apply(3.0, 4.0)
        self.assertTrue(approx(x, 6.0))
        self.assertTrue(approx(y, 2.0))

    def test_rotation_quarter_turn(self):
        t = AffineTransform.rotation(math.pi / 2)
        x, y = t.apply(1.0, 0.0)
        self.assertTrue(approx(x, 0.0))
        self.assertTrue(approx(y, 1.0))

    def test_rotation_degrees(self):
        t = AffineTransform.identity().rotate_degrees(90.0)
        x, y = t.apply(1.0, 0.0)
        self.assertTrue(approx(x, 0.0))
        self.assertTrue(approx(y, 1.0))

    def test_shear(self):
        t = AffineTransform.shearing(1.0, 0.0)
        x, y = t.apply(2.0, 3.0)
        self.assertTrue(approx(x, 5.0))
        self.assertTrue(approx(y, 3.0))


class TestChaining(unittest.TestCase):
    def test_chain_returns_new_instance(self):
        a = AffineTransform.translation(1.0, 2.0)
        b = a.translate(3.0, 4.0)
        self.assertIsNot(a, b)
        # original unchanged
        self.assertTrue(approx(a.matrix()[0][2], 1.0))

    def test_translate_then_rotate_origin(self):
        # rotate first, then translate: point (0,0) just translates.
        t = AffineTransform.translation(100.0, 50.0).rotate(math.pi / 2)
        x, y = t.apply(0.0, 0.0)
        self.assertTrue(approx(x, 100.0))
        self.assertTrue(approx(y, 50.0))

    def test_rotate_then_translate_point(self):
        # translate first, then rotate: (1,0) translates to (101,50),
        # then rotates a quarter turn to (-50,101).
        t = AffineTransform.rotation(math.pi / 2).translate(100.0, 50.0)
        x, y = t.apply(1.0, 0.0)
        self.assertTrue(approx(x, -50.0))
        self.assertTrue(approx(y, 101.0))

    def test_translate_rotate_scale_compose(self):
        t = AffineTransform().translate(10.0, 0.0).rotate(math.pi / 2).scale(2.0, 2.0)
        # order: scale (1,0)->(2,0); rotate -> (0,2); translate -> (10,2)
        x, y = t.apply(1.0, 0.0)
        self.assertTrue(approx(x, 10.0))
        self.assertTrue(approx(y, 2.0))

    def test_associativity(self):
        a = AffineTransform.translation(1.0, 2.0)
        b = AffineTransform.scaling(2.0, 3.0)
        c = AffineTransform.rotation(0.5)
        left = (a._chain(b))._chain(c)
        right = a._chain((b._chain(c)))
        self.assertEqual(left, right)


class TestEquality(unittest.TestCase):
    def test_equal_within_eps(self):
        a = AffineTransform([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        b = AffineTransform([[1.0 + 1e-15, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(a, b)

    def test_not_equal_different(self):
        a = AffineTransform.translation(1.0, 0.0)
        b = AffineTransform.translation(2.0, 0.0)
        self.assertNotEqual(a, b)

    def test_hash_consistent_with_eq(self):
        a = AffineTransform([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        b = AffineTransform([[1.0 + 1e-15, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(hash(a), hash(b))


class TestMatrix(unittest.TestCase):
    def test_matrix_is_copy(self):
        t = AffineTransform.translation(1.0, 2.0)
        m = t.matrix()
        m[0][2] = 999.0
        self.assertTrue(approx(t.matrix()[0][2], 1.0))

    def test_repr_roundtrips(self):
        t = AffineTransform.translation(1.0, 2.0)
        r = repr(t)
        self.assertIn("AffineTransform", r)
        self.assertIn("1.0", r)


if __name__ == "__main__":
    unittest.main()
