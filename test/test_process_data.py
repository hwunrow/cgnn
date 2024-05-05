import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import torch_geometric

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.process_data import (
    create_torch_geometric_temporal_data,
    create_edge_indices,
    create_edge_weights,
    create_features_targets,
    create_torch_geometric_data,
    create_train_test_mask,
    create_edge_index,
    process_case_death_data,
)


class TestFunctions(unittest.TestCase):

    def test_create_torch_geometric_temporal_data(self):
        data = create_torch_geometric_temporal_data()
        for snapshot in data:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (5, 22)
            assert snapshot.y.shape == (5,)

    def test_create_edge_indices(self):
        dates = [datetime(2020, 2, 29), datetime(2020, 3, 1), datetime(2020, 3, 2)]
        edge_indices = create_edge_indices(dates)
        self.assertEqual(len(edge_indices), len(dates))
        self.assertIsInstance(edge_indices[0], np.ndarray)

    def test_create_edge_weights(self):
        dates = [datetime(2020, 2, 29), datetime(2020, 3, 1), datetime(2020, 3, 2)]
        edge_snapshot = create_edge_indices([1])[0]
        edge_weights = create_edge_weights(dates, edge_snapshot)
        self.assertEqual(len(edge_weights), len(dates))
        self.assertIsInstance(edge_weights[0], np.ndarray)
        self.assertEqual(len(edge_weights[0]), edge_snapshot.shape[1])

    def test_create_features_targets(self):
        dates = [datetime(2020, 2, 29), datetime(2020, 3, 1)]
        node_dict = {
            "36005-2020-02-29": 0,
            "36047-2020-02-29": 1,
            "36061-2020-02-29": 2,
            "36081-2020-02-29": 3,
            "36085-2020-02-29": 4,
            "36005-2020-03-01": 5,
            "36047-2020-03-01": 6,
            "36061-2020-03-01": 7,
            "36081-2020-03-01": 8,
            "36085-2020-03-01": 9,
        }
        features, targets = create_features_targets(dates, node_dict)
        self.assertEqual(len(features), len(dates))
        self.assertEqual(len(targets), len(dates))
        self.assertIsInstance(features[0], np.ndarray)
        self.assertIsInstance(targets[0], np.ndarray)
        self.assertEqual(features[0].shape, (5, 22))
        self.assertEqual(targets[0].shape, (5,))

    def test_create_torch_geometric_data(self):
        # Test if create_torch_geometric_data returns a torch_geometric Data object
        data = create_torch_geometric_data("cpu")
        self.assertTrue(isinstance(data, torch_geometric.data.Data))

    def test_create_train_test_mask(self):
        # Test if create_train_test_mask returns two lists of the correct length
        node_dict = {"36061-2020-03-01": 0, "36061-2020-03-02": 1}  # Example node_dict
        dates = [
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2020-03-02"),
        ]  # Example dates
        fips_list = [36061]  # Example FIPS list
        train_mask, test_mask = create_train_test_mask(node_dict, dates, fips_list)
        self.assertEqual(len(train_mask), len(node_dict))
        self.assertEqual(len(test_mask), len(node_dict))

    def test_create_edge_index(self):
        # Test if create_edge_index returns a pandas DataFrame with the correct shape
        node_dict = {"36061-2020-03-01": 0, "36061-2020-03-02": 1}  # Example node_dict
        dates = [
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2020-03-02"),
        ]  # Example dates
        fips_list = [36061]  # Example FIPS list
        coo_df = create_edge_index(node_dict, dates, fips_list)
        self.assertTrue(isinstance(coo_df, pd.DataFrame))
        self.assertEqual(
            coo_df.shape[1], 2
        )  # Expecting two columns for source and target indices

    def test_process_case_death_data(self):
        # Test if process_case_death_data returns two pandas DataFrames
        death_subset_df, case_subset_df = process_case_death_data()
        self.assertTrue(isinstance(death_subset_df, pd.DataFrame))
        self.assertTrue(isinstance(case_subset_df, pd.DataFrame))

        test_date = "03/25/2020"
        death_truth = {
            36005: np.mean([7, 13, 14, 12, 14, 14, 35]),
            36047: np.mean([6, 10, 14, 10, 28, 28, 44]),
            36061: np.mean([4, 12, 6, 9, 10, 19, 24]),
            36081: np.mean([6, 10, 14, 14, 36, 40, 36]),
            36085: np.mean([4, 4, 3, 7, 8, 6, 4]),
        }
        test_death_df = death_subset_df.loc[
            case_subset_df.date_of_interest == test_date,
            ["FIPS", "DEATH_COUNT_7DAY_AVG"],
        ]
        test_death_dict = dict(
            zip(test_death_df["FIPS"], test_death_df["DEATH_COUNT_7DAY_AVG"])
        )
        self.assertDictEqual(test_death_dict, death_truth)

        case_truth = {
            36005: np.mean([623, 723, 491, 495, 730, 927, 1068]),
            36047: np.mean([1204, 1136, 554, 755, 909, 1212, 1255]),
            36061: np.mean([555, 653, 399, 317, 530, 619, 586]),
            36081: np.mean([1065, 1185, 947, 696, 1192, 1389, 1570]),
            36085: np.mean([258, 310, 248, 317, 209, 350, 394]),
        }
        case_delta_truth = {
            36005: (998.0 - 623.0) / 7,
            36047: (1372.0 - 1204.0) / 7,
            36061: (625.0 - 555.0) / 7,
            36081: (1644.0 - 1065.0) / 7,
            36085: (405.0 - 258.0) / 7,
        }
        test_case_df = case_subset_df.loc[
            case_subset_df.date_of_interest == test_date,
            ["FIPS", "CASE_COUNT_7DAY_AVG", "CASE_DELTA"],
        ]
        test_case_dict = dict(
            zip(test_case_df["FIPS"], test_case_df["CASE_COUNT_7DAY_AVG"])
        )
        test_case_df = case_subset_df.loc[
            case_subset_df.date_of_interest == test_date,
            ["FIPS", "CASE_COUNT_7DAY_AVG", "CASE_DELTA"],
        ]
        test_case_dict = dict(
            zip(test_case_df["FIPS"], test_case_df["CASE_COUNT_7DAY_AVG"])
        )
        test_case_delta_dict = dict(
            zip(test_case_df["FIPS"], test_case_df["CASE_DELTA"])
        )
        self.assertDictEqual(test_case_dict, case_truth)
        self.assertDictEqual(
            {key: round(test_case_delta_dict[key], 3) for key in test_case_delta_dict},
            {key: round(case_delta_truth[key], 3) for key in case_delta_truth},
        )


if __name__ == "__main__":
    unittest.main()
