FROM mambaorg/micromamba:1.3.1 as mamba

USER root
SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/root/.cache/apt apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates wget curl git \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' >> .bashrc \
    && echo 'export PATH=$HOME/bin:$HOME/.local/bin:$PATH' >> .bashrc \
    && echo 'source /opt/conda/etc/profile.d/conda.sh' >> .bashrc

WORKDIR /tmp
COPY environment.yaml requirements.txt /tmp/

RUN --mount=type=cache,target=/root/.cache/mamba eval "$(micromamba shell hook --shell=bash)" \
    && micromamba create -y --file environment.yaml


FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04
SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/root/.cache/apt apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=mamba /bin/micromamba /bin/micromamba
COPY --from=mamba /opt/conda/envs/best-rq/ /opt/conda/envs/best-rq/
COPY --from=mamba /tmp/ /tmp/

COPY . .

ENV PATH=/opt/conda/envs/best-rq/bin/:$PATH

RUN export MAMBA_ROOT_PREFIX=/opt/conda/ \
    && eval "$(micromamba shell hook --shell=bash)" \
    && micromamba activate best-rq
