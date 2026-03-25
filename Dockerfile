# This adds a conditional build step. Use `--build-arg GPU=1` to build with CUDA/CUDNN support,
# omit it to only support CPU.  

ARG GPU=0

FROM debian:bookworm-slim AS base-cpu
FROM base-cpu AS build-0

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base-gpu
FROM base-gpu AS build-1

# Conditional CPU/GPU image selection 
FROM build-${GPU} AS final

# Docker ARG's don't survive FROM statements, hence we need to declare it a second time. 
ARG GPU
ARG TARGETARCH
ARG TARGETVARIANT

# Install faster-whisper
WORKDIR /usr/src

COPY ./pyproject.toml ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    \
    && python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && if [ "$GPU" = "1" ]; then \
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"; \
    else \
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"; \
    fi \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url "$TORCH_INDEX" \
        'torch==2.6.0' \
    \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url https://www.piwheels.org/simple \
        -e '.[zeroconf,transformers,sherpa,onnx-asr]' \
    \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

EXPOSE 10400

ENTRYPOINT ["bash", "docker_run.sh"]
