FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev build-essential \
    ffmpeg tmux psmisc netcat-openbsd ca-certificates git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Create user
ARG USER=carlauser
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $USER && useradd -m -u $UID -g $GID -s /bin/bash $USER
RUN mkdir -p /workspace/base/carla_project && chown -R $USER:$USER /workspace

WORKDIR /workspace/base/carla_project
USER $USER
SHELL ["/bin/bash","-lc"]

# Copy only code + configs, not old venv
COPY --chown=$USER:$USER CarDreamer ./CarDreamer
COPY --chown=$USER:$USER carla ./carla
COPY --chown=$USER:$USER env.sh requirements.txt ./

# Create venv + install deps
RUN python3 -m venv venv \
 && source venv/bin/activate \
 && pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "jax[cuda12]==0.4.34" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Env vars
ENV PYTHONNOUSERSITE=1
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Auto-source env
RUN echo 'source /workspace/base/carla_project/env.sh' >> /home/$USER/.bashrc

WORKDIR /workspace/base/carla_project/CarDreamer
CMD ["/bin/bash"]
