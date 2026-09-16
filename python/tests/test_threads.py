"""Threads: the GIL is released during heavy calls, concurrent use from Python threads is safe,
and num_threads does not change results."""

import threading
import time

import numpy as np

import vectorforge as vf


def test_parallel_search_matches_serial():
    rng = np.random.default_rng(1)
    xb = rng.standard_normal((2000, 32), dtype=np.float32)
    xq = rng.standard_normal((200, 32), dtype=np.float32)
    idx = vf.Index(dim=32, index="flat")
    idx.add(xb, num_threads=4)
    serial = idx.search(xq, k=10, num_threads=1)
    parallel = idx.search(xq, k=10, num_threads=0)
    np.testing.assert_array_equal(serial[0], parallel[0])
    np.testing.assert_array_equal(serial[1], parallel[1])


def test_gil_is_released_during_search():
    rng = np.random.default_rng(2)
    xb = rng.standard_normal((20000, 64), dtype=np.float32)
    idx = vf.Index(dim=64, index="flat")
    idx.add(xb)
    xq = rng.standard_normal((400, 64), dtype=np.float32)
    ticks = 0
    done = threading.Event()

    def ticker():
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    start = time.perf_counter()
    idx.search(xq, k=10, num_threads=1)
    elapsed = time.perf_counter() - start
    done.set()
    t.join()
    # With the GIL held for the whole search, the ticker could not run at all.
    if elapsed > 0.05:
        assert ticks >= 5, (ticks, elapsed)


def test_concurrent_python_threads():
    rng = np.random.default_rng(3)
    idx = vf.Index(dim=16, M=8, ef_construction=40)
    idx.add(rng.standard_normal((500, 16), dtype=np.float32))
    errors = []

    def writer(base):
        local_rng = np.random.default_rng(base)  # Generators are not thread-safe
        try:
            for i in range(20):
                idx.add(local_rng.standard_normal((10, 16), dtype=np.float32),
                        ids=np.arange(base + i * 10, base + i * 10 + 10), num_threads=2)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def reader():
        try:
            for _ in range(50):
                labels, distances = idx.search(np.ones((4, 16), dtype=np.float32), k=5)
                assert (np.diff(distances, axis=1) >= 0).all()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(10_000 * (w + 1),)) for w in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(idx) == 500 + 2 * 200
