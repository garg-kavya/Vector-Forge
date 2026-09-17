// `vectorforge serve`: runs the HTTP API over a catalog directory until SIGINT/SIGTERM
// (Ctrl+C, console close or shutdown on Windows).

#include <atomic>
#include <chrono>
#include <csignal>
#include <ostream>
#include <thread>

#include "commands.hpp"
#include "server/server.hpp"
#include "util/log.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vf::cli {

namespace {

std::atomic<bool> g_stop_requested{false};

void on_signal(int /*signal*/) {
  g_stop_requested.store(true);
}

#if defined(_WIN32)
BOOL WINAPI on_console_event(DWORD event) {
  switch (event) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
      g_stop_requested.store(true);
      return TRUE;
    default:
      return FALSE;
  }
}
#endif

void install_handlers() {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
#if defined(_WIN32)
  SetConsoleCtrlHandler(on_console_event, TRUE);
#endif
}

}  // namespace

void request_serve_stop() noexcept {
  g_stop_requested.store(true);
}

Status run_serve(const ServeOptions& options, std::ostream& log) {
  g_stop_requested.store(false);
  LoadOptions load;
  load.use_mmap = options.use_mmap;
  Result<std::unique_ptr<Catalog>> catalog = Catalog::open(options.data_dir, load);
  if (!catalog.ok()) {
    return catalog.status();
  }
  server::Server server(*catalog.value(), options.server);
  VF_RETURN_IF_ERROR(server.start());
  if (options.install_signal_handlers) {
    install_handlers();
  }
  log << "vectorforge serving " << catalog.value()->size() << " collection(s) from "
      << options.data_dir.string() << " on http://" << options.server.host << ":" << server.port()
      << '\n'
      << std::flush;
  vf::log::write(vf::log::Level::Info, "server_started",
                 {{"host", options.server.host},
                  {"port", server.port()},
                  {"collections", catalog.value()->size()},
                  {"auth", !options.server.api_key.empty()}});
  while (!g_stop_requested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  log << "shutting down\n" << std::flush;
  vf::log::write(vf::log::Level::Info, "server_stopping");
  server.stop();
  if (options.snapshot_on_exit) {
    const Status snapshots = catalog.value()->snapshot_all();
    if (!snapshots.ok()) {
      vf::log::write(vf::log::Level::Error, "snapshot_failed", {{"error", snapshots.to_string()}});
      return snapshots;
    }
    log << "snapshots written\n" << std::flush;
    vf::log::write(vf::log::Level::Info, "snapshots_written");
  }
  return {};
}

}  // namespace vf::cli
