// VectorForge C++ quickstart: create a collection, insert, search, remove, save and reload.
//
// Built against an installed VectorForge by tests/install/consumer:
//   find_package(vectorforge REQUIRED)
//   target_link_libraries(app PRIVATE vectorforge::vectorforge)

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <vectorforge/vectorforge.hpp>

namespace {

int fail(const vf::Status& status) {
  std::fprintf(stderr, "error: %s\n", status.to_string().c_str());
  return EXIT_FAILURE;
}

}  // namespace

int main() {
  vf::CollectionConfig config;
  config.dim = 4;
  config.metric = vf::Metric::Cosine;  // vectors are normalised on insert and query
  config.index = vf::IndexType::Hnsw;
  config.hnsw.M = 16;
  config.hnsw.ef_construction = 200;

  vf::Result<std::unique_ptr<vf::Collection>> created = vf::Collection::create(config);
  if (!created.ok()) {
    return fail(created.status());
  }
  std::unique_ptr<vf::Collection> collection = std::move(created).value();

  // A batch: ids and row-major vectors.
  const std::vector<vf::ExternalId> ids{7, 42, 99};
  const std::vector<float> rows{
      1.0F, 0.0F, 0.0F, 0.0F,  //
      0.6F, 0.8F, 0.0F, 0.0F,  //
      0.0F, 0.0F, 1.0F, 0.0F,
  };
  vf::ThreadPool pool(2);  // optional: parallel insertion and batch search
  if (vf::Result<std::size_t> added = collection->add_batch(ids, rows, {}, &pool); !added.ok()) {
    return fail(added.status());
  }

  vf::SearchParams params;
  params.k = 2;
  params.ef_search = 64;
  const std::vector<float> query{0.5F, 0.9F, 0.0F, 0.0F};
  vf::Result<std::vector<vf::Neighbor>> hits = collection->search(query, params);
  if (!hits.ok()) {
    return fail(hits.status());
  }
  std::printf("nearest: %llu (distance %.4f), simd tier %s\n",
              static_cast<unsigned long long>(hits.value()[0].id),
              static_cast<double>(hits.value()[0].distance),
              std::string(vf::to_string(vf::active_simd_level())).c_str());

  if (vf::Status st = collection->remove(99); !st.ok()) {
    return fail(st);
  }

  const std::filesystem::path file = "quickstart.vfidx";
  if (vf::Status st = collection->save(file); !st.ok()) {
    return fail(st);
  }
  vf::Result<std::unique_ptr<vf::Collection>> loaded = vf::Collection::load(file);
  if (!loaded.ok()) {
    return fail(loaded.status());
  }
  std::printf("reloaded %zu vectors (version %s)\n", loaded.value()->size(),
              std::string(vf::kVersion).c_str());
  loaded.value().reset();  // release the file mapping before deleting the file
  std::filesystem::remove(file);
  return EXIT_SUCCESS;
}
