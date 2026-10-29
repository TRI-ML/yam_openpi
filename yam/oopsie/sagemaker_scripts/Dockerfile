# openpi (dsrl_openpi_flip fork, YAM policies/configs) training image for SageMaker, launched by the yam_eval
# Train tab (yam_eval/train/compute.py, family pi05). Modeled on the DROID fork's scripts/sagemaker image:
# everything arrives via environment variables + input channels, no hyperparameter parsing (see entrypoint.sh).
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates clang curl git git-lfs ffmpeg linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /opt/ml/code
RUN uv venv --python 3.11 $UV_PROJECT_ENVIRONMENT
COPY pyproject.toml uv.lock ./
COPY packages/openpi-client/pyproject.toml packages/openpi-client/pyproject.toml
COPY packages/openpi-client/src packages/openpi-client/src
RUN --mount=type=cache,target=/root/.cache/uv GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen
# uv.lock resolves torchcodec to a CUDA-13 build (wants libnvrtc.so.13) while torch is 2.7.1+cu126: LeRobot's
# video decode then fails (preflight 2026-09-20). Pin the torchcodec that matches torch 2.7 / cu126.
RUN uv pip install --python /opt/venv --no-deps "torchcodec==0.4.0" --index-url https://download.pytorch.org/whl/cu126
ENV PATH="/opt/venv/bin:${PATH}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OPENPI_DATA_HOME=/opt/openpi_cache HF_HOME=/opt/hf_cache
RUN python -c "import openpi.shared.download as download; download.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}); print('tokenizer prefetched')" \
    && test -f /opt/openpi_cache/big_vision/paligemma_tokenizer.model
ARG OPENPI_GIT_SHA=unknown
ENV OPENPI_GIT_SHA=${OPENPI_GIT_SHA}
RUN chmod +x /opt/ml/code/scripts/sagemaker/entrypoint.sh
ENTRYPOINT ["/opt/ml/code/scripts/sagemaker/entrypoint.sh"]
