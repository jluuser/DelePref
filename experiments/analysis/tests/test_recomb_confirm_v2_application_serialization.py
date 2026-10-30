import json
import unittest

import numpy as np

from recomb_confirm_v2_application_serialization import json_native


class SerializationTests(unittest.TestCase):
    def test_nested_numpy_scalars_round_trip_without_value_change(self):
        original = {'counts': [np.int64(2**53+1), np.int32(0)],
                    'bounds': (np.float64(.125), np.float32(.5)),
                    'known': np.bool_(False), 'unknown': None}
        converted = json_native(original)
        saved = json.loads(json.dumps(converted, allow_nan=False))
        self.assertEqual(saved, {'counts': [2**53+1, 0], 'bounds': [.125, .5],
                                 'known': False, 'unknown': None})
        self.assertIs(type(converted['counts'][0]), int)
        self.assertIs(type(converted['known']), bool)

    def test_native_results_remain_identical(self):
        original = {'N_all': 80, 'mean_lower': 0., 'mean_upper': None,
                    'nested': [{'status': 'FAILED', 'coverage': .75}]}
        self.assertEqual(json_native(original), original)

    def test_nonfinite_values_are_still_rejected(self):
        with self.assertRaises(ValueError):
            json.dumps(json_native({'x': np.float64(np.nan)}), allow_nan=False)

    def test_numpy_arrays_are_not_silently_reinterpreted(self):
        with self.assertRaises(TypeError):
            json.dumps(json_native({'x': np.array([1, 2])}), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
