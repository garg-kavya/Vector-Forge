# VectorForge server image (docs/http-api.md, "Docker").
#
#   docker build -t vectorforge .
#   docker run --rm -p 8080:8080 -v vf-data:/data vectorforge
#
# Build stage: portable x86-64 baseline code with the runtime-dispatched AVX2 kernels, so the
# image runs on any x86-64 host.
FROM ubuntu:24.04 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++-14 cmake ninja-build ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++-14 \
      -DVF_BUILD_TESTS=OFF -DVF_BUILD_BENCHMARKS=OFF -DVF_BUILD_CLI=ON -DVF_BUILD_SERVER=ON \
 && cmake --build /build --target vectorforge_cli \
 && strip /build/apps/cli/vectorforge

# Runtime stage: the binary, its C++ runtime, and an unprivileged user.
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home-dir /data vectorforge \
 && mkdir -p /data && chown vectorforge /data
COPY --from=build /build/apps/cli/vectorforge /usr/local/bin/vectorforge
USER vectorforge
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1
# Binds all interfaces inside the container; publish the port deliberately. Set VF_API_KEY to
# require a bearer token.
ENTRYPOINT ["vectorforge", "serve", "--data-dir", "/data", "--host", "0.0.0.0", "--port", "8080"]
