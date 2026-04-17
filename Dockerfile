FROM debian:bookworm-slim
ARG TARGETARCH
ARG TARGETVARIANT
ARG SHERPA_ONNX_GPU_WHEEL="1.12.20+cuda12.cudnn9"

# Install faster-whisper
WORKDIR /usr/src

COPY ./pyproject.toml ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libasound2 \
        python3 \
        python3-pip \
        python3-venv \
    \
    && python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url 'https://download.pytorch.org/whl/cpu' \
        'torch==2.6.0' \
    \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url https://www.piwheels.org/simple \
        -e '.[zeroconf,transformers,sherpa,onnx-asr]' \
    && .venv/bin/pip3 uninstall -y sherpa-onnx sherpa-onnx-core \
    && .venv/bin/pip3 install --no-cache-dir \
        "sherpa-onnx==${SHERPA_ONNX_GPU_WHEEL}" \
        -f https://k2-fsa.github.io/sherpa/onnx/cuda.html \
    \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

EXPOSE 10400

ENTRYPOINT ["bash", "docker_run.sh"]
