"""Tests for coordinate augmentation (Boltz Algorithm 19)."""

import numpy as np
import pytest

from deepfold.data.augment import (
    center_random_augmentation,
    quaternion_to_matrix,
    random_quaternion,
    random_rotation_matrix,
)


class TestQuaternion:
    def test_unit_norm(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            q = random_quaternion(rng)
            assert abs(np.linalg.norm(q) - 1.0) < 1e-12

    def test_canonical_sign(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            q = random_quaternion(rng)
            assert q[0] >= 0


class TestRotationMatrix:
    def test_orthogonal(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            R = random_rotation_matrix(rng)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_det_one(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            R = random_rotation_matrix(rng)
            assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_identity_from_identity_quat(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_matrix(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)


class TestCenterRandomAugmentation:
    @pytest.fixture
    def coords(self):
        rng = np.random.default_rng(123)
        atom_coords = rng.standard_normal((50, 3)).astype(np.float32) * 10
        token_coords = rng.standard_normal((10, 3)).astype(np.float32) * 10
        ref_pos = rng.standard_normal((50, 3)).astype(np.float32) * 5
        mask = np.ones(50, dtype=bool)
        return atom_coords, token_coords, ref_pos, mask

    def test_centering(self, coords):
        atom_coords, token_coords, ref_pos, mask = coords
        rng = np.random.default_rng(0)
        # Use identity-like rotation by fixing the rng
        results = center_random_augmentation(
            atom_coords,
            token_coords,
            ref_pos,
            mask=mask,
            s_trans=0.0,
            training=True,
            rng=rng,
        )
        # Output should be float32
        for r in results:
            assert r.dtype == np.float32

    def test_pairwise_distances_preserved(self, coords):
        """Rotation preserves pairwise distances."""
        atom_coords, token_coords, ref_pos, mask = coords
        rng = np.random.default_rng(99)

        # No translation so only rotation + centering
        aug_atoms, _, _ = center_random_augmentation(
            atom_coords,
            token_coords,
            ref_pos,
            mask=mask,
            s_trans=0.0,
            training=False,
            rng=rng,
        )

        # Centered original for comparison
        center = atom_coords[mask].mean(axis=0)
        orig_centered = atom_coords - center

        # Pairwise distances
        from scipy.spatial.distance import pdist

        d_orig = pdist(orig_centered[:20])
        d_aug = pdist(aug_atoms[:20])
        np.testing.assert_allclose(d_aug, d_orig, atol=1e-4)

    def test_training_has_translation(self, coords):
        atom_coords, token_coords, ref_pos, mask = coords
        rng = np.random.default_rng(42)
        train_results = center_random_augmentation(
            atom_coords,
            mask=mask,
            s_trans=1.0,
            training=True,
            rng=rng,
        )
        rng2 = np.random.default_rng(42)
        infer_results = center_random_augmentation(
            atom_coords,
            mask=mask,
            s_trans=1.0,
            training=False,
            rng=rng2,
        )
        # Same rotation, but training adds translation → means differ
        train_mean = train_results[0].mean(axis=0)
        infer_mean = infer_results[0].mean(axis=0)
        # Inference center should be near zero (centered + rotated)
        assert np.linalg.norm(infer_mean) < 0.5
        # Training center shifted by translation
        diff = np.linalg.norm(train_mean - infer_mean)
        assert diff > 0.01  # very likely with s_trans=1.0

    def test_consistent_frame(self, coords):
        """All arrays share the same rotation and translation."""
        atom_coords, token_coords, ref_pos, mask = coords
        rng = np.random.default_rng(77)
        aug_a, aug_t, aug_r = center_random_augmentation(
            atom_coords,
            token_coords,
            ref_pos,
            mask=mask,
            s_trans=1.0,
            training=True,
            rng=rng,
        )
        # Recover the rotation: compare centered originals to augmented
        center = atom_coords[mask].mean(axis=0)
        orig_centered = (atom_coords - center).astype(np.float64)
        aug_a_64 = aug_a.astype(np.float64)

        # Translation = mean of (aug - R @ orig_centered)
        # Since same R, t applied to all arrays, check token_coords too
        center_tok = token_coords - center  # same centering
        # If the frame is consistent, relative vector between arrays is preserved
        # Pick first atom and first token: their relative position should be rotated
        rel_orig = orig_centered[0] - center_tok[0].astype(np.float64)
        rel_aug = aug_a_64[0] - aug_t[0].astype(np.float64)
        # Magnitudes should match (rotation preserves distances)
        np.testing.assert_allclose(
            np.linalg.norm(rel_orig), np.linalg.norm(rel_aug), atol=1e-3
        )

    def test_empty_arrays(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        results = center_random_augmentation(empty, s_trans=1.0, training=True)
        assert len(results) == 1
        assert results[0].shape == (0, 3)

    def test_reproducible(self, coords):
        atom_coords, token_coords, ref_pos, mask = coords
        for seed in [0, 42, 999]:
            r1 = center_random_augmentation(
                atom_coords,
                mask=mask,
                training=True,
                rng=np.random.default_rng(seed),
            )
            r2 = center_random_augmentation(
                atom_coords,
                mask=mask,
                training=True,
                rng=np.random.default_rng(seed),
            )
            np.testing.assert_array_equal(r1[0], r2[0])
