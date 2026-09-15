import unittest

import numpy as np

from src.data import make_split


class DataProtocolTests(unittest.TestCase):
    def test_high_load_split_is_ordered_and_disjoint(self):
        split = make_split("high_load")
        self.assertTrue(set(split.train_idx).isdisjoint(split.val_idx))
        self.assertTrue(set(split.train_idx).isdisjoint(split.test_idx))
        self.assertTrue(set(split.val_idx).isdisjoint(split.test_idx))
        self.assertLessEqual(split.y_train_raw.max(), split.y_val_raw.min())
        self.assertLessEqual(split.y_val_raw.max(), split.y_test_raw.min())

    def test_scalers_are_fit_on_training_only(self):
        split = make_split("high_load")
        np.testing.assert_allclose(split.scaler_x.mean_, split.x_train_raw.mean(axis=0), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(split.scaler_y.mean_[0], split.y_train_raw.mean(), rtol=1e-10, atol=1e-10)

    def test_random_split_expected_sizes_and_disjointness(self):
        split = make_split("random")
        self.assertEqual(
            len(split.train_idx) + len(split.val_idx) + len(split.test_idx),
            split.metadata["n_total"],
        )
        self.assertTrue(set(split.train_idx).isdisjoint(split.val_idx))
        self.assertTrue(set(split.train_idx).isdisjoint(split.test_idx))
        self.assertTrue(set(split.val_idx).isdisjoint(split.test_idx))


if __name__ == "__main__":
    unittest.main()
