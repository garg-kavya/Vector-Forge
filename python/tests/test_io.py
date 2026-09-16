"""Persistence and input conversion: save/load in both modes, zero-copy for float32 C-contiguous
input, one conversion otherwise, strict mode."""

import numpy as np
import pytest

import vectorforge as vf


def make_index(n=300, dim=8, index="hnsw"):
    rng = np.random.default_rng(3)
    xb = rng.standard_normal((n, dim), dtype=np.float32)
    idx = vf.Index(dim=dim, metric="l2", index=index, seed=11)
    idx.add(xb, np.arange(n, dtype=np.uint64) * 3)
    return idx, xb


@pytest.mark.parametrize("mmap", [True, False])
def test_save_and_load_round_trip(tmp_path, mmap):
    idx, xb = make_index()
    idx.remove([6])
    path = tmp_path / "idx.vfidx"
    idx.save(path)
    loaded = vf.Index.load(str(path), mmap=mmap, verify="full")
    assert len(loaded) == len(idx) == 299
    assert loaded.config == idx.config
    np.testing.assert_array_equal(loaded.get(9), xb[3])
    a = idx.search(xb[:20], k=5, ef_search=64)
    b = loaded.search(xb[:20], k=5, ef_search=64)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    # Loaded indexes keep accepting vectors.
    loaded.add(np.zeros((1, 8), dtype=np.float32), ids=[10**9])
    assert 10**9 in loaded
    del loaded  # release a mapping before tmp_path cleanup (Windows)


def test_load_errors(tmp_path):
    with pytest.raises(OSError):
        vf.Index.load(tmp_path / "missing.vfidx")
    bad = tmp_path / "bad.vfidx"
    bad.write_bytes(b"not an index" * 10)
    with pytest.raises(vf.CorruptIndexError):
        vf.Index.load(bad)
    with pytest.raises(ValueError):
        vf.Index.load(bad, verify="sometimes")
    idx, _ = make_index(n=20)
    good = tmp_path / "good.vfidx"
    idx.save(good)
    data = bytearray(good.read_bytes())
    data[-100] ^= 0xFF
    good.write_bytes(bytes(data))
    with pytest.raises(vf.CorruptIndexError):
        vf.Index.load(good, mmap=False, verify="full")


def test_float32_c_contiguous_input_is_not_copied():
    idx = vf.Index(dim=4, index="flat")
    xb = np.ones((10, 4), dtype=np.float32)
    idx.add(xb)
    assert vf._vectorforge._last_input_address() == xb.ctypes.data
    q = np.ones((3, 4), dtype=np.float32)
    idx.search(q, k=1)
    assert vf._vectorforge._last_input_address() == q.ctypes.data


def test_other_inputs_are_converted_once_or_rejected_in_strict_mode():
    idx = vf.Index(dim=4, index="flat")
    as_float64 = np.ones((5, 4), dtype=np.float64)
    assert idx.add(as_float64) == 5
    assert vf._vectorforge._last_input_address() != as_float64.ctypes.data
    fortran = np.asfortranarray(np.ones((5, 4), dtype=np.float32))
    assert idx.add(fortran) == 5
    assert idx.add([[1, 2, 3, 4]]) == 1  # nested lists work too
    labels, _ = idx.search([1.0, 2.0, 3.0, 4.0], k=1)
    assert labels.shape == (1,)
    with pytest.raises(TypeError):
        idx.add(as_float64, strict=True)
    with pytest.raises(TypeError):
        idx.search(fortran, k=1, strict=True)
    idx.search(np.ones((1, 4), dtype=np.float32), k=1, strict=True)


def test_shape_and_value_errors():
    idx = vf.Index(dim=4)
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        idx.add(np.full((1, 4), np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 4), dtype=np.float32), ids=[1])
    with pytest.raises(ValueError):
        idx.add(np.ones((1, 4), dtype=np.float32), ids=[-1])
    with pytest.raises(TypeError):
        idx.add(np.ones((1, 4), dtype=np.float32), ids=[1.5])
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 4), dtype=np.float32), ids=[3, 3])
    with pytest.raises(ValueError):
        idx.search(np.ones(4, dtype=np.float32), k=0)
    with pytest.raises(ValueError):
        idx.search(np.ones(4, dtype=np.float32), k=5, ef_search=0)
    with pytest.raises(ValueError):
        idx.add(np.ones((1, 4), dtype=np.float32), num_threads=-1)
    with pytest.raises(TypeError):
        idx.add("not an array")
    assert len(idx) == 0, "failed calls insert nothing"
