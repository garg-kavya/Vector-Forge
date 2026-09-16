"""Exception mapping (docs/python-api.md, "Errors")."""

import numpy as np
import pytest

import vectorforge as vf


def test_hierarchy():
    assert issubclass(vf.DuplicateIdError, vf.VectorForgeError)
    assert issubclass(vf.DuplicateIdError, ValueError)
    assert issubclass(vf.CorruptIndexError, vf.VectorForgeError)
    assert not issubclass(vf.CorruptIndexError, ValueError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": 4, "metric": "hamming"},
        {"dim": 4, "index": "ivf"},
        {"dim": 4, "M": 1},
        {"dim": 4, "concurrency": "maybe"},
    ],
)
def test_invalid_configuration_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        vf.Index(**kwargs)


def test_missing_ids_raise_key_error():
    idx = vf.Index(dim=2)
    with pytest.raises(KeyError):
        idx.get(1)
    with pytest.raises(KeyError):
        idx.remove([1])


def test_zero_vector_in_cosine_index():
    idx = vf.Index(dim=3, metric="cosine")
    with pytest.raises(ValueError):
        idx.add(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        idx.search(np.zeros(3, dtype=np.float32), k=1)


def test_save_to_unwritable_path(tmp_path):
    idx = vf.Index(dim=2)
    with pytest.raises(OSError):
        idx.save(tmp_path / "no" / "such" / "dir" / "x.vfidx")
