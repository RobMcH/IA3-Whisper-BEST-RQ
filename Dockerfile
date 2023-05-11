FROM mambaorg/micromamba:1.4.2 as mamba

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
USER root
SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/root/.cache/apt apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=mamba /bin/micromamba /bin/micromamba
COPY --from=mamba /opt/conda/envs/best-rq/ /root/micromamba/envs/best-rq/
COPY --from=mamba /tmp/ /tmp/

COPY . .

ENV PATH=/root/micromamba/envs/best-rq/bin/:$PATH
ENV MAMBA_ROOT_PREFIX=/root/micromamba/
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN micromamba shell init --shell=bash \
    && echo "micromamba activate best-rq" >> /root/.bashrc \
    && python -m pip install -e .
