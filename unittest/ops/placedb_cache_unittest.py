##
# @file   placedb_cache_unittest.py
#

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
from dreamplace import PlaceDB
sys.path.pop()


class PlaceDBCacheTest(unittest.TestCase):
    def test_safe_cache_codec_round_trip(self):
        value = {
            "array": np.arange(6, dtype=np.float32).reshape(2, 3),
            "numpy_scalar": np.int32(7),
            "numpy_type": np.float64,
            "bytes": b"\x00DREAMPlace\xff",
            "mapping": {"node \u03b1": 0, "node \u03b2": np.int32(1)},
            "nested": [None, True, (1.5, "value")],
        }
        arrays = {}
        encoded = PlaceDB.PlaceDB._database_cache_encode(value, arrays)
        manifest = json.dumps(encoded, sort_keys=True, separators=(",", ":"))

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = os.path.join(temp_dir, "cache.npz")
            np.savez(cache_file, manifest=np.asarray(manifest), **arrays)
            with np.load(cache_file, allow_pickle=False) as archive:
                decoded = PlaceDB.PlaceDB._database_cache_decode(
                    json.loads(str(archive["manifest"].item())), archive
                )

        np.testing.assert_array_equal(decoded["array"], value["array"])
        self.assertIsInstance(decoded["numpy_scalar"], np.int32)
        self.assertEqual(decoded["numpy_scalar"], value["numpy_scalar"])
        self.assertIs(decoded["numpy_type"], np.float64)
        self.assertEqual(decoded["bytes"], value["bytes"])
        self.assertEqual(decoded["mapping"], {"node \u03b1": 0, "node \u03b2": 1})
        self.assertEqual(decoded["nested"], value["nested"])

    def test_object_arrays_are_rejected(self):
        arrays = {}
        with self.assertRaises(TypeError):
            PlaceDB.PlaceDB._database_cache_encode(
                np.asarray([object()], dtype=object), arrays
            )
        self.assertEqual(arrays, {})


if __name__ == "__main__":
    unittest.main()
