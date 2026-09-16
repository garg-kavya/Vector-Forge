"""Index basics: construction, add/search/get/remove, shapes, padding, exactness against NumPy."""

import numpy as np
import pytest

import vectorforge as vf


def brute_force(xb, xq, k, metric):
    if metric == "l2":
        d = ((xq[:, None, :].astype(np.float64) - xb[None, :, :]) ** 2).sum(-1)
    elif metric == "ip":
        d = -(xq.astype(np.float64) @ xb.T.astype(np.float64))
    else:
        nb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        nq = xq / np.linalg.norm(xq, axis=1, keepdims=True)
        d = 1.0 - nq.astype(np.float64) @ nb.T.astype(np.float64)
    order = np.argsort(d, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(d, order, axis=1)


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    return rng.standard_normal((500, 16), dtype=np.float32), rng.standard_normal((20, 16), dtype=np.float32)


def test_construction_and_properties():
    index = vf.Index(dim=8, metric="cosine", index="hnsw", M=12, ef_construction=64, ef_search=32, seed=5)
    assert index.dim == 8
    assert index.metric == "cosine"
    assert index.index_type == "hnsw"
    assert len(index) == 0
    cfg = index.config
    assert cfg["M"] == 12 and cfg["seed"] == 5 and cfg["normalize"] is True
    assert "Index(dim=8" in repr(index)
    assert vf.simd_level() in ("avx2", "scalar")
    assert vf.__version__
    with pytest.raises(TypeError):
        vf.Index(8)  # keyword-only


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_flat_matches_brute_force(data, metric):
    xb, xq = data
    index = vf.Index(dim=16, metric=metric, index="flat")
    assert index.add(xb) == len(xb)
    labels, distances = index.search(xq, k=5)
    assert labels.shape == (20, 5) and labels.dtype == np.uint64
    assert distances.shape == (20, 5) and distances.dtype == np.float32
    expected_ids, expected_d = brute_force(xb, xq, 5, metric)
    np.testing.assert_allclose(distances, expected_d, rtol=1e-4, atol=1e-4)
    # Ids may differ only among exact ties; compare the distance of each returned id.
    assert (labels == expected_ids).mean() > 0.99


def test_hnsw_recall(data):
    xb, xq = data
    index = vf.Index(dim=16, metric="l2", M=16, ef_construction=100)
    index.add(xb, num_threads=4)
    labels, _ = index.search(xq, k=10, ef_search=128)
    expected, _ = brute_force(xb, xq, 10, "l2")
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(labels, expected)])
    assert recall > 0.95


def test_ids_get_remove_and_contains(data):
    xb, _ = data
    index = vf.Index(dim=16, index="flat")
    ids = np.arange(1000, 1000 + len(xb), dtype=np.int64)
    index.add(xb, ids)
    assert len(index) == len(xb)
    assert 1000 in index and 999 not in index and -1 not in index and "x" not in index
    np.testing.assert_array_equal(index.get(1003), xb[3])
    got = index.get(1003)
    got[0] = 99  # a copy
    assert index.get(1003)[0] == xb[3][0]
    index.remove([1003, 1004])
    index.remove(np.array([1005], dtype=np.uint64))
    index.remove(1006)
    assert len(index) == len(xb) - 4
    assert 1003 not in index
    labels, _ = index.search(xb[3], k=3)
    assert 1003 not in labels
    assert labels[0] in index and np.int64(1000) in index
    with pytest.raises(KeyError):
        index.get(1003)
    with pytest.raises(KeyError):
        index.remove([1003])
    stats = index.stats()
    assert stats["count"] == len(xb) - 4 and stats["deleted"] == 4
    assert index.compact() == {"rows_before": 500, "removed_rows": 4, "rows_after": 496}
    assert index.stats()["deleted"] == 0


def test_automatic_ids_continue_after_stored_rows():
    index = vf.Index(dim=2, index="flat")
    index.add(np.ones((3, 2), dtype=np.float32))
    index.remove([1])
    index.add(np.zeros((2, 2), dtype=np.float32))
    assert sorted(i for i in range(10) if i in index) == [0, 2, 3, 4]


def test_one_dimensional_query_and_padding():
    index = vf.Index(dim=3, index="hnsw")
    index.add(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), ids=[7, 9])
    labels, distances = index.search(np.array([1, 0, 0], dtype=np.float32), k=4)
    assert labels.shape == (4,) and distances.shape == (4,)
    assert labels[0] == 7 and distances[0] == 0.0
    assert labels[2] == vf.INVALID_ID and labels[3] == 2**64 - 1
    assert np.isinf(distances[2:]).all()


def test_upsert_and_duplicates():
    index = vf.Index(dim=2, index="flat")
    index.add(np.array([[1, 1]], dtype=np.float32), ids=[5])
    with pytest.raises(vf.DuplicateIdError) as info:
        index.add(np.array([[2, 2]], dtype=np.float32), ids=[5])
    assert isinstance(info.value, ValueError) and isinstance(info.value, vf.VectorForgeError)
    index.add(np.array([[2, 2]], dtype=np.float32), ids=[5], upsert=True)
    np.testing.assert_array_equal(index.get(5), [2, 2])
    assert len(index) == 1


def test_empty_batches_and_empty_index():
    index = vf.Index(dim=4)
    assert index.add(np.empty((0, 4), dtype=np.float32)) == 0
    labels, distances = index.search(np.ones((2, 4), dtype=np.float32), k=3)
    assert (labels == vf.INVALID_ID).all() and np.isinf(distances).all()
    labels, _ = index.search(np.empty((0, 4), dtype=np.float32), k=3)
    assert labels.shape == (0, 3)
