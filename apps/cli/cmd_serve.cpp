// `vectorforge serve`: runs the HTTP API over a catalog directory until SIGINT/SIGTERM
// (Ctrl+C, console close or shutdown on Windows).

#include <atomic>
#include <chrono>
#include <csignal>
#include <ostream>
#include <thread>

#include "commands.hpp"
#include "server/server.hpp"

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
  while (!g_stop_requested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  log << "shutting down\n" << std::flush;
  server.stop();
  if (options.snapshot_on_exit) {
    VF_RETURN_IF_ERROR(catalog.value()->snapshot_all());
    log << "snapshots written\n" << std::flush;
  }
  return {};
}

}  // namespace vf::cli
