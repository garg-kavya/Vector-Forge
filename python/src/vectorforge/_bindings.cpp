// Python bindings (docs/python-api.md, docs/DESIGN.md §14).
//
// Copies: float32 C-contiguous inputs are read in place; anything else is converted once by NumPy
// (or rejected with strict=True). add() copies each row into the index once. search() writes into
// NumPy result arrays allocated here. Long calls release the GIL; the input arrays stay alive
// through the argument references.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <vectorforge/collection.hpp>
#include <vectorforge/simd.hpp>
#include <vectorforge/thread_pool.hpp>
#include <vectorforge/version.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// Data pointer of the last float array handed to the library (tests check zero-copy with it).
std::atomic<std::uintptr_t> g_last_input{0};

// Exceptions (created in the module init).
PyObject* g_error = nullptr;  // vectorforge.VectorForgeError
PyObject* g_duplicate_error =
    nullptr;                          // vectorforge.DuplicateIdError(VectorForgeError, ValueError)
PyObject* g_corrupt_error = nullptr;  // vectorforge.CorruptIndexError(VectorForgeError)

[[noreturn]] void raise(const vf::Status& status) {
  const std::string message = status.message();
  switch (status.code()) {
    case vf::ErrorCode::InvalidArgument:
    case vf::ErrorCode::DimensionMismatch:
      throw py::value_error(message);
    case vf::ErrorCode::NotFound:
      throw py::key_error(message);
    case vf::ErrorCode::AlreadyExists:
      PyErr_SetString(g_duplicate_error, message.c_str());
      throw py::error_already_set();
    case vf::ErrorCode::IoError:
      PyErr_SetString(PyExc_OSError, message.c_str());
      throw py::error_already_set();
    case vf::ErrorCode::CorruptData:
    case vf::ErrorCode::UnsupportedVersion:
      PyErr_SetString(g_corrupt_error, message.c_str());
      throw py::error_already_set();
    case vf::ErrorCode::ResourceExhausted:
      PyErr_SetString(PyExc_MemoryError, message.c_str());
      throw py::error_already_set();
    default:
      PyErr_SetString(g_error,
                      (std::string(vf::to_string(status.code())) + ": " + message).c_str());
      throw py::error_already_set();
  }
}

void check(const vf::Status& status) {
  if (!status.ok()) {
    raise(status);
  }
}

template <class T>
T unwrap(vf::Result<T> result) {
  if (!result.ok()) {
    raise(result.status());
  }
  return std::move(result).value();
}

// Shared pools per thread count (thread creation per call would dominate small batches).
std::shared_ptr<vf::ThreadPool> pool_for(int num_threads) {
  if (num_threads < 0) {
    throw py::value_error("num_threads must be >= 0");
  }
  const std::size_t hw = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
  const std::size_t threads = num_threads == 0 ? hw : static_cast<std::size_t>(num_threads);
  if (threads <= 1) {
    return nullptr;
  }
  static std::mutex mutex;
  static std::map<std::size_t, std::shared_ptr<vf::ThreadPool>> pools;
  const std::lock_guard<std::mutex> lock(mutex);
  std::shared_ptr<vf::ThreadPool>& pool = pools[threads];
  if (!pool) {
    pool = std::make_shared<vf::ThreadPool>(threads - 1);
  }
  return pool;
}

// Rows of a 1-D (d,) or 2-D (n, d) float array.
struct Rows {
  FloatArray array;
  std::size_t count = 0;
  bool one_dimensional = false;
};

Rows as_rows(const py::handle& obj, std::uint32_t dim, bool strict, const char* what) {
  if (strict) {
    const auto plain = py::array::ensure(obj);
    if (!plain || !py::isinstance<py::array_t<float>>(plain) ||
        (plain.flags() & py::array::c_style) == 0) {
      throw py::type_error(std::string(what) +
                           " must be a C-contiguous float32 array when strict=True");
    }
  }
  Rows rows;
  rows.array = FloatArray::ensure(obj);
  if (!rows.array) {
    throw py::type_error(std::string(what) + " must be convertible to a float32 array");
  }
  if (rows.array.ndim() == 1) {
    rows.one_dimensional = true;
    rows.count = 1;
    if (static_cast<std::size_t>(rows.array.shape(0)) != dim) {
      throw py::value_error(std::string(what) + " has " + std::to_string(rows.array.shape(0)) +
                            " components, expected " + std::to_string(dim));
    }
  } else if (rows.array.ndim() == 2) {
    rows.count = static_cast<std::size_t>(rows.array.shape(0));
    if (static_cast<std::size_t>(rows.array.shape(1)) != dim) {
      throw py::value_error(std::string(what) + " has " + std::to_string(rows.array.shape(1)) +
                            " columns, expected " + std::to_string(dim));
    }
  } else {
    throw py::value_error(std::string(what) + " must be 1-D (d,) or 2-D (n, d)");
  }
  g_last_input.store(reinterpret_cast<std::uintptr_t>(rows.array.data()));
  return rows;
}

// External ids from any integer array-like (negative values are rejected before conversion).
std::vector<vf::ExternalId> as_ids(const py::handle& obj, std::size_t expected) {
  py::array plain = py::array::ensure(obj);
  if (!plain) {
    throw py::type_error("ids must be an array of integers");
  }
  if (plain.ndim() == 0) {
    plain = plain.attr("reshape")(1);
  }
  const char kind = plain.dtype().kind();
  if (kind != 'i' && kind != 'u') {
    throw py::type_error("ids must have an integer dtype");
  }
  if (kind == 'i' && plain.size() > 0 && py::cast<bool>(plain.attr("min")().attr("__lt__")(0))) {
    throw py::value_error("ids must be non-negative");
  }
  const auto ids = py::array_t < std::uint64_t,
             py::array::c_style | py::array::forcecast > ::ensure(plain.attr("ravel")());
  if (static_cast<std::size_t>(ids.size()) != expected) {
    throw py::value_error("got " + std::to_string(ids.size()) + " ids for " +
                          std::to_string(expected) + " vectors");
  }
  return {ids.data(), ids.data() + ids.size()};
}

vf::Metric metric_from(const std::string& name) {
  return unwrap(vf::parse_metric(name));
}

vf::Verify verify_from(const std::string& name) {
  if (name == "auto") {
    return vf::Verify::Auto;
  }
  if (name == "none") {
    return vf::Verify::None;
  }
  if (name == "metadata") {
    return vf::Verify::Metadata;
  }
  if (name == "full") {
    return vf::Verify::Full;
  }
  throw py::value_error("verify must be 'auto', 'none', 'metadata' or 'full'");
}

class PyIndex {
 public:
  PyIndex(std::uint32_t dim, const std::string& metric, const std::string& index, std::uint32_t m,
          std::uint32_t ef_construction, std::uint32_t ef_search, std::uint64_t seed,
          bool normalize, const std::string& concurrency) {
    vf::CollectionConfig cfg;
    cfg.dim = dim;
    cfg.metric = metric_from(metric);
    cfg.index = unwrap(vf::parse_index_type(index));
    cfg.normalize = normalize;
    cfg.concurrency = unwrap(vf::parse_concurrency(concurrency));
    cfg.hnsw.M = m;
    cfg.hnsw.ef_construction = ef_construction;
    cfg.hnsw.ef_search = ef_search;
    cfg.hnsw.seed = seed;
    collection_ = unwrap(vf::Collection::create(cfg));
  }

  explicit PyIndex(std::unique_ptr<vf::Collection> collection)
      : collection_(std::move(collection)) {}

  static PyIndex load(const std::filesystem::path& path, bool mmap, const std::string& verify,
                      const std::string& concurrency) {
    vf::LoadOptions options;
    options.use_mmap = mmap;
    options.verify = verify_from(verify);
    options.concurrency = unwrap(vf::parse_concurrency(concurrency));
    vf::Result<std::unique_ptr<vf::Collection>> loaded = vf::Status::internal("not run");
    {
      const py::gil_scoped_release release;
      loaded = vf::Collection::load(path, options);
    }
    return PyIndex(unwrap(std::move(loaded)));
  }

  std::size_t add(const py::handle& vectors, const py::object& ids, bool upsert, int num_threads,
                  bool strict) {
    const Rows rows = as_rows(vectors, dim(), strict, "vectors");
    std::vector<vf::ExternalId> labels;
    if (ids.is_none()) {
      // Automatic ids are row numbers: they continue after every row ever stored (including
      // removed ones), so they never collide with other automatic ids.
      const std::uint64_t first = collection_->stats().row_count;
      labels.resize(rows.count);
      for (std::size_t i = 0; i < rows.count; ++i) {
        labels[i] = first + i;
      }
    } else {
      labels = as_ids(ids, rows.count);
    }
    const std::shared_ptr<vf::ThreadPool> pool = pool_for(num_threads);
    const std::span<const float> data(rows.array.data(), rows.count * dim());
    vf::Result<std::size_t> inserted = vf::Status::internal("not run");
    {
      const py::gil_scoped_release release;
      inserted = collection_->add_batch(labels, data, {.upsert = upsert}, pool.get());
    }
    return unwrap(std::move(inserted));
  }

  py::tuple search(const py::handle& queries, std::uint32_t k, std::optional<std::uint32_t> ef,
                   int num_threads, bool strict) const {
    if (k == 0) {
      throw py::value_error("k must be >= 1");
    }
    const Rows rows = as_rows(queries, dim(), strict, "queries");
    vf::SearchParams params;
    params.k = k;
    params.ef_search = ef;
    check(params.validate());
    std::vector<py::ssize_t> shape;
    if (rows.one_dimensional) {
      shape = {static_cast<py::ssize_t>(k)};
    } else {
      shape = {static_cast<py::ssize_t>(rows.count), static_cast<py::ssize_t>(k)};
    }
    py::array_t<std::uint64_t> labels(shape);
    py::array_t<float> distances(shape);
    std::vector<std::uint32_t> counts(rows.count);
    const std::shared_ptr<vf::ThreadPool> pool = pool_for(num_threads);
    const std::span<const float> data(rows.array.data(), rows.count * dim());
    const std::span<vf::ExternalId> out_ids(labels.mutable_data(),
                                            static_cast<std::size_t>(labels.size()));
    const std::span<float> out_dist(distances.mutable_data(),
                                    static_cast<std::size_t>(distances.size()));
    vf::Status st;
    {
      const py::gil_scoped_release release;
      st = collection_->search_batch(data, rows.count, params, out_ids, out_dist, counts,
                                     pool.get());
    }
    check(st);
    return py::make_tuple(labels, distances);
  }

  void remove(const py::object& ids) {
    const py::array plain = py::array::ensure(ids);
    const std::size_t n = plain ? static_cast<std::size_t>(plain.size()) : 0;
    for (const vf::ExternalId id : as_ids(ids, n)) {
      check(collection_->remove(id));
    }
  }

  py::array_t<float> get(vf::ExternalId id) const {
    const std::vector<float> v = unwrap(collection_->get(id));
    py::array_t<float> out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
  }

  bool contains(const py::object& id) const {
    // Anything with __index__ (Python and NumPy integers) is an id candidate.
    const auto value = py::reinterpret_steal<py::object>(PyNumber_Index(id.ptr()));
    if (!value) {
      PyErr_Clear();
      return false;
    }
    if (py::cast<bool>(value.attr("__lt__")(0)) ||
        py::cast<bool>(value.attr("__ge__")(py::int_(vf::kInvalidExternalId)))) {
      return false;
    }
    return collection_->contains(py::cast<vf::ExternalId>(value));
  }

  void save(const std::filesystem::path& path) const {
    vf::Status st;
    {
      const py::gil_scoped_release release;
      st = collection_->save(path);
    }
    check(st);
  }

  py::dict compact() {
    vf::Result<vf::CompactStats> stats = vf::Status::internal("not run");
    {
      const py::gil_scoped_release release;
      stats = collection_->compact();
    }
    const vf::CompactStats s = unwrap(std::move(stats));
    py::dict out;
    out["rows_before"] = s.rows_before;
    out["removed_rows"] = s.removed_rows;
    out["rows_after"] = s.rows_after;
    return out;
  }

  py::dict stats() const {
    const vf::CollectionStats s = collection_->stats();
    py::dict memory;
    memory["vectors_bytes"] = s.memory.vectors_bytes;
    memory["mapped_vectors_bytes"] = s.memory.mapped_vectors_bytes;
    memory["labels_bytes"] = s.memory.labels_bytes;
    memory["id_map_bytes_estimate"] = s.memory.id_map_bytes_estimate;
    memory["tombstone_bytes"] = s.memory.tombstone_bytes;
    memory["index_bytes"] = s.memory.index_bytes;
    memory["total_bytes"] = s.memory.total_bytes();
    py::dict out;
    out["count"] = s.live_count;
    out["deleted"] = s.deleted_count;
    out["rows"] = s.row_count;
    out["dim"] = s.dim;
    out["metric"] = std::string(vf::to_string(s.metric));
    out["index"] = std::string(vf::to_string(s.index));
    out["normalized"] = s.normalized;
    out["simd"] = std::string(vf::to_string(s.simd));
    out["memory"] = memory;
    return out;
  }

  py::dict config() const {
    const vf::CollectionConfig& c = collection_->config();
    py::dict out;
    out["dim"] = c.dim;
    out["metric"] = std::string(vf::to_string(c.metric));
    out["normalize"] = c.effective_normalize();
    out["index"] = std::string(vf::to_string(c.index));
    out["M"] = c.hnsw.M;
    out["ef_construction"] = c.hnsw.ef_construction;
    out["ef_search"] = c.hnsw.ef_search;
    out["max_level"] = c.hnsw.max_level;
    out["seed"] = c.hnsw.seed;
    out["concurrency"] = std::string(vf::to_string(c.concurrency));
    return out;
  }

  [[nodiscard]] std::uint32_t dim() const { return collection_->config().dim; }
  [[nodiscard]] std::string metric() const {
    return std::string(vf::to_string(collection_->config().metric));
  }
  [[nodiscard]] std::string index_type() const {
    return std::string(vf::to_string(collection_->config().index));
  }
  [[nodiscard]] std::size_t size() const { return collection_->size(); }

 private:
  std::shared_ptr<vf::Collection> collection_;
};

}  // namespace

PYBIND11_MODULE(_vectorforge, m) {
  m.doc() = "VectorForge: exact and HNSW vector similarity search (C++ core).";

  // The exception types live as long as the module (references intentionally not released).
  g_error =
      PyErr_NewExceptionWithDoc("vectorforge._vectorforge.VectorForgeError",
                                "Base class of VectorForge errors.", PyExc_Exception, nullptr);
  if (g_error == nullptr) {
    throw py::error_already_set();
  }
  m.attr("VectorForgeError") = py::handle(g_error);
  const py::tuple duplicate_bases =
      py::make_tuple(py::handle(g_error), py::handle(PyExc_ValueError));
  g_duplicate_error =
      PyErr_NewExceptionWithDoc("vectorforge._vectorforge.DuplicateIdError",
                                "An id that already exists was inserted without upsert=True.",
                                duplicate_bases.ptr(), nullptr);
  g_corrupt_error = PyErr_NewExceptionWithDoc(
      "vectorforge._vectorforge.CorruptIndexError",
      "An index file is damaged or has an unsupported format version.", g_error, nullptr);
  if (g_duplicate_error == nullptr || g_corrupt_error == nullptr) {
    throw py::error_already_set();
  }
  m.attr("DuplicateIdError") = py::handle(g_duplicate_error);
  m.attr("CorruptIndexError") = py::handle(g_corrupt_error);

  m.attr("__version__") = std::string(vf::kVersion);
  m.attr("git_sha") = std::string(vf::kGitSha);
  m.attr("INVALID_ID") = py::int_(vf::kInvalidExternalId);

  m.def(
      "simd_level",
      [] {
        check(vf::simd_status());
        return std::string(vf::to_string(vf::active_simd_level()));
      },
      "Kernel tier in use: 'avx2' or 'scalar'.");
  m.def(
      "_last_input_address", [] { return g_last_input.load(); },
      "Address of the last float array passed to the library (testing zero-copy).");

  py::class_<PyIndex>(m, "Index")
      .def(py::init<std::uint32_t, const std::string&, const std::string&, std::uint32_t,
                    std::uint32_t, std::uint32_t, std::uint64_t, bool, const std::string&>(),
           py::kw_only(), py::arg("dim"), py::arg("metric") = "l2", py::arg("index") = "hnsw",
           py::arg("M") = 16, py::arg("ef_construction") = 200, py::arg("ef_search") = 50,
           py::arg("seed") = 0x5EEDF0A6EULL, py::arg("normalize") = false,
           py::arg("concurrency") = "concurrent")
      .def_static("load", &PyIndex::load, py::arg("path"), py::kw_only(), py::arg("mmap") = true,
                  py::arg("verify") = "auto", py::arg("concurrency") = "concurrent")
      .def("add", &PyIndex::add, py::arg("vectors"), py::arg("ids") = py::none(), py::kw_only(),
           py::arg("upsert") = false, py::arg("num_threads") = 0, py::arg("strict") = false)
      .def("search", &PyIndex::search, py::arg("queries"), py::kw_only(), py::arg("k") = 10,
           py::arg("ef_search") = py::none(), py::arg("num_threads") = 0, py::arg("strict") = false)
      .def("remove", &PyIndex::remove, py::arg("ids"))
      .def("get", &PyIndex::get, py::arg("id"))
      .def("save", &PyIndex::save, py::arg("path"))
      .def("compact", &PyIndex::compact)
      .def("stats", &PyIndex::stats)
      .def_property_readonly("config", &PyIndex::config)
      .def_property_readonly("dim", &PyIndex::dim)
      .def_property_readonly("metric", &PyIndex::metric)
      .def_property_readonly("index_type", &PyIndex::index_type)
      .def("__len__", &PyIndex::size)
      .def("__contains__", &PyIndex::contains)
      .def("__repr__", [](const PyIndex& self) {
        return "Index(dim=" + std::to_string(self.dim()) + ", metric='" + self.metric() +
               "', index='" + self.index_type() + "', size=" + std::to_string(self.size()) + ")";
      });
}
