# yam_openpi (TRI-ML/yam_openpi) SageMaker training image for pi05 YAM pretraining (ABC-130k + MolmoAct2). Build from the repo root:
#   docker build -f yam/sagemaker/Dockerfile -t <ecr>/anthonyliang-openpi:<tag> .
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 UV_LINK_MODE=copy UV_PYTHON=3.11 PATH=/root/.local/bin:/opt/tools:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git xz-utils unzip tmux procps jq rsync htop build-essential linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
# tools: uv, static ffmpeg/ffprobe (libdav1d for AV1), s5cmd, awscli v2
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mkdir -p /opt/tools && cd /opt/tools \
    && curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ff.tar.xz \
    && tar -xJf ff.tar.xz --strip-components=1 --wildcards '*/ffmpeg' '*/ffprobe' && rm ff.tar.xz \
    && curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz s5cmd \
    && curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip && cd /tmp && unzip -q awscli.zip && ./aws/install && rm -rf /tmp/aws /tmp/awscli.zip
# code + venv (uv.lock pins jax 0.5.3 cuda12, torch 2.7.1, lerobot git 0cf86487 = v2.1, pyav w/ dav1d)
COPY . /opt/ml/code/yam_openpi
WORKDIR /opt/ml/code/yam_openpi
RUN uv python install 3.11 && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev && rm -rf /root/.cache/uv && .venv/bin/python -c "import etils, orbax, jax, lerobot; print('venv ok', jax.__version__)"
# sanity: configs import, pyav has dav1d, ffmpeg present
RUN .venv/bin/python -c "import sys; sys.path.insert(0,'src'); from openpi.training import config as C; C.get_config('pi05_yam_abc_mm2_h32'); C.get_config('pi05_yam_abc_mm2_h32_smoke'); print('configs ok')" \
    && .venv/bin/python -c "import av, jax; print('pyav', av.__version__, 'dav1d' if any('dav1d' in c for c in av.codecs_available) else 'NO-DAV1D')" \
    && ffmpeg -hide_banner -version | head -1 && chmod +x yam/sagemaker/*.sh yam/pretrain/*.sh yam/multihost/*.sh
ENV PYTHONUNBUFFERED=1 SAGEMAKER_PROGRAM=/opt/ml/code/yam_openpi/yam/sagemaker/sm_entry.sh
ENTRYPOINT ["/opt/ml/code/yam_openpi/yam/sagemaker/sm_entry.sh"]
