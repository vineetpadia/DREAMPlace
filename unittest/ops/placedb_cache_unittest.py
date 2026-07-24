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
    def test_derived_compatibility_maps_are_lazy(self):
        db = PlaceDB.PlaceDB()
        db.node_names = np.asarray([b"node0", b"node1"])
        db.net_names = np.asarray([b"net0", b"net1"])
        db.flat_node2pin_map = np.asarray([0, 2, 1], dtype=np.int32)
        db.flat_node2pin_start_map = np.asarray([0, 2, 3], dtype=np.int32)
        db.flat_net2pin_map = np.asarray([1, 0, 2], dtype=np.int32)
        db.flat_net2pin_start_map = np.asarray([0, 1, 3], dtype=np.int32)
        db._node_name2id_map = None
        db._net_name2id_map = None

        self.assertEqual(db.num_nets, 2)
        self.assertIsNone(db._node_name2id_map)
        self.assertIsNone(db._net_name2id_map)
        self.assertIsNone(db._node2pin_map)
        self.assertIsNone(db._net2pin_map)

        self.assertEqual(db.node_name2id_map, {"node0": 0, "node1": 1})
        self.assertEqual(db.net_name2id_map, {"net0": 0, "net1": 1})
        np.testing.assert_array_equal(db.node2pin_map[0], [0, 2])
        np.testing.assert_array_equal(db.node2pin_map[1], [1])
        np.testing.assert_array_equal(db.net2pin_map[0], [1])
        np.testing.assert_array_equal(db.net2pin_map[1], [0, 2])

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
