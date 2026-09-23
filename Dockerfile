# Dockerfile to build environemnt for CREDIT
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python  \
    PATH=/opt/venv/bin:${PATH} 

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        make \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install ${PYTHON_VERSION} \
    && uv venv --python ${PYTHON_VERSION} ${VIRTUAL_ENV}

RUN uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 \
    && uv pip install matplotlib numpy

WORKDIR /workspace

ENV USER=credit \
    LOGNAME=credit

# The repository is bind-mounted at /workspace when the container starts.
CMD ["bash"]
