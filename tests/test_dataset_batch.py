"""Tests for batched collation with padding and mask generation."""

from __future__ import annotations

import torch

from deepfold.data.dataset import collate_fn


def _make_sample(
    N: int,
    N_atom: int,
    n_pairs: int,
    N_prot: int = 0,
    training: bool = True,
) -> dict[str, torch.Tensor]:
    """Create a synthetic feature dict mimicking featurize() output.

    Parameters match the shapes documented in featurize.py.
    """
    if N_prot == 0:
        N_prot = N  # default: all tokens are protein

    features: dict[str, torch.Tensor] = {
        "token_type": torch.randint(0, 4, (N,)),
        "profile": torch.randn(N, 32),
        "del_mean": torch.randn(N, 1),
        "has_msa": torch.ones(N, 1),
        "msa_feat": torch.randn(1, 1, N_prot, 34),
        "c_atom": torch.randn(N_atom, 197),
        "p_lm": torch.randn(n_pairs, 5),
        "p_lm_idx": torch.randint(0, max(N_atom, 1), (n_pairs, 2)),
        "token_idx": torch.randint(0, max(N, 1), (N_atom,)),
        "chain_id": torch.zeros(N, dtype=torch.int64),
        "global_idx": torch.arange(N, dtype=torch.int64),
        "bond_matrix": torch.zeros(N, N, dtype=torch.bool),
        "msa_token_mask": torch.ones(N, dtype=torch.bool),
        # Padding masks (all-ones for real data)
        "token_pad_mask": torch.ones(N, dtype=torch.float32),
        "atom_pad_mask": torch.ones(N_atom, dtype=torch.float32),
        "pair_valid_mask": torch.ones(n_pairs, dtype=torch.float32),
    }

    if training:
        features["x_atom_true"] = torch.randn(N_atom, 3)
        features["x_res_true"] = torch.randn(N, 3)
        features["atom_resolved_mask"] = torch.ones(N_atom, dtype=torch.float32)
        features["token_resolved_mask"] = torch.ones(N, dtype=torch.float32)

    return features


class TestCollateBatchSize1:
    """batch_size=1 adds a batch dimension via unsqueeze(0)."""

    def test_single_sample_batch_dim(self):
        sample = _make_sample(N=10, N_atom=50, n_pairs=20)
        result = collate_fn([sample])
        # Should add batch dim (B=1) to all tensors
        assert result["token_pad_mask"].shape == (1, 10)
        assert result["atom_pad_mask"].shape == (1, 50)
        assert result["bond_matrix"].shape == (1, 10, 10)

    def test_single_sample_values_preserved(self):
        sample = _make_sample(N=10, N_atom=50, n_pairs=20)
        result = collate_fn([sample])
        # Values should be identical (just unsqueezed)
        torch.testing.assert_close(result["profile"][0], sample["profile"])


class TestCollateBatch:
    """batch_size>1 pads and stacks correctly."""

    def test_basic_shapes(self):
        """Two samples of different sizes produce correct batch shapes."""
        s1 = _make_sample(N=10, N_atom=50, n_pairs=20, N_prot=10)
        s2 = _make_sample(N=15, N_atom=80, n_pairs=35, N_prot=15)
        result = collate_fn([s1, s2])

        B = 2
        max_N = 15
        max_N_atom = 80
        max_n_pairs = 35
        max_N_prot = 15

        assert result["token_type"].shape == (B, max_N)
        assert result["profile"].shape == (B, max_N, 32)
        assert result["c_atom"].shape == (B, max_N_atom, 197)
        assert result["p_lm"].shape == (B, max_n_pairs, 5)
        assert result["p_lm_idx"].shape == (B, max_n_pairs, 2)
        assert result["token_idx"].shape == (B, max_N_atom)
        assert result["bond_matrix"].shape == (B, max_N, max_N)
        assert result["msa_feat"].shape == (B, 1, 1, max_N_prot, 34)

    def test_mask_values(self):
        """Mask tensors have 1.0 for real data and 0.0 for padding."""
        s1 = _make_sample(N=8, N_atom=30, n_pairs=10, N_prot=8)
        s2 = _make_sample(N=12, N_atom=50, n_pairs=25, N_prot=12)
        result = collate_fn([s1, s2])

        # token_pad_mask: s1 has N=8, padded to 12
        assert result["token_pad_mask"].shape == (2, 12)
        assert result["token_pad_mask"][0, :8].sum() == 8.0  # all real
        assert result["token_pad_mask"][0, 8:].sum() == 0.0  # all padding
        assert result["token_pad_mask"][1, :12].sum() == 12.0  # all real

        # atom_pad_mask: s1 has N_atom=30, padded to 50
        assert result["atom_pad_mask"].shape == (2, 50)
        assert result["atom_pad_mask"][0, :30].sum() == 30.0
        assert result["atom_pad_mask"][0, 30:].sum() == 0.0
        assert result["atom_pad_mask"][1, :50].sum() == 50.0

        # pair_valid_mask
        assert result["pair_valid_mask"].shape == (2, 25)
        assert result["pair_valid_mask"][0, :10].sum() == 10.0
        assert result["pair_valid_mask"][0, 10:].sum() == 0.0

    def test_bool_pad_with_false(self):
        """Boolean tensors (bond_matrix) are padded with False."""
        s1 = _make_sample(N=5, N_atom=10, n_pairs=5, N_prot=5)
        s2 = _make_sample(N=8, N_atom=15, n_pairs=10, N_prot=8)
        # Set a bond in s1
        s1["bond_matrix"][0, 1] = True
        s1["bond_matrix"][1, 0] = True

        result = collate_fn([s1, s2])
        assert result["bond_matrix"].dtype == torch.bool
        assert result["bond_matrix"].shape == (2, 8, 8)
        # Original bond preserved
        assert (
            result["bond_matrix"][0, 0, 1] is True
            or result["bond_matrix"][0, 0, 1].item()
        )
        assert (
            result["bond_matrix"][0, 1, 0] is True
            or result["bond_matrix"][0, 1, 0].item()
        )
        # Padded region is False
        assert not result["bond_matrix"][0, 5:, :].any()
        assert not result["bond_matrix"][0, :, 5:].any()

    def test_real_data_preserved(self):
        """Original tensor values are preserved in padded batch."""
        s1 = _make_sample(N=6, N_atom=20, n_pairs=8, N_prot=6)
        s2 = _make_sample(N=10, N_atom=40, n_pairs=15, N_prot=10)
        result = collate_fn([s1, s2])

        # Check that s1's profile data is preserved
        torch.testing.assert_close(result["profile"][0, :6, :], s1["profile"])
        # Check s2's c_atom is preserved
        torch.testing.assert_close(result["c_atom"][1, :40, :], s2["c_atom"])

    def test_zero_pad_coordinates(self):
        """Coordinate tensors (x_atom_true, x_res_true) are zero-padded."""
        s1 = _make_sample(N=5, N_atom=15, n_pairs=5, N_prot=5)
        s2 = _make_sample(N=8, N_atom=25, n_pairs=10, N_prot=8)
        result = collate_fn([s1, s2])

        # s1 has N_atom=15, padded to 25
        assert result["x_atom_true"].shape == (2, 25, 3)
        # Padded region should be zeros
        assert result["x_atom_true"][0, 15:, :].abs().sum() == 0.0
        # Real data preserved
        torch.testing.assert_close(result["x_atom_true"][0, :15, :], s1["x_atom_true"])

    def test_equal_sizes_no_padding(self):
        """Samples with identical sizes still batch correctly."""
        s1 = _make_sample(N=10, N_atom=40, n_pairs=20, N_prot=10)
        s2 = _make_sample(N=10, N_atom=40, n_pairs=20, N_prot=10)
        result = collate_fn([s1, s2])

        assert result["token_pad_mask"].shape == (2, 10)
        # All mask values should be 1.0 (no padding needed)
        assert result["token_pad_mask"].sum() == 20.0
        assert result["atom_pad_mask"].sum() == 80.0

    def test_msa_pad_mask_generated(self):
        """msa_pad_mask is generated for batches with different MSA sizes."""
        s1 = _make_sample(N=8, N_atom=20, n_pairs=5, N_prot=6)
        s2 = _make_sample(N=12, N_atom=35, n_pairs=10, N_prot=10)
        result = collate_fn([s1, s2])

        assert "msa_pad_mask" in result
        # Shape: (B, max_N_prot)
        assert result["msa_pad_mask"].shape == (2, 10)
        # s1 has N_prot=6, s2 has N_prot=10
        assert result["msa_pad_mask"][0, :6].sum() == 6.0
        assert result["msa_pad_mask"][0, 6:].sum() == 0.0
        assert result["msa_pad_mask"][1, :10].sum() == 10.0

    def test_three_samples(self):
        """Batching 3 samples of different sizes works."""
        s1 = _make_sample(N=5, N_atom=10, n_pairs=3, N_prot=5)
        s2 = _make_sample(N=12, N_atom=50, n_pairs=20, N_prot=12)
        s3 = _make_sample(N=8, N_atom=30, n_pairs=10, N_prot=8)
        result = collate_fn([s1, s2, s3])

        assert result["token_type"].shape[0] == 3
        assert result["token_type"].shape[1] == 12  # max N
        assert result["c_atom"].shape[1] == 50  # max N_atom
        assert result["p_lm"].shape[1] == 20  # max n_pairs
