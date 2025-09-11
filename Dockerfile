# CUDA 12.9.0 + cuDNN8 + Ubuntu 22.04
# 다른 태그는: https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
# ⚠️ PyTorch 사전빌드 휠은 cu118/cu121/cu124 위주입니다.
# cu129 휠이 없다면 torch 버전을 cu124에 맞추거나(권장) 소스 빌드가 필요합니다.
ARG PYTORCH='2.4.1'
ARG TORCH_CUDA='cu124'   # cu129 미제공 시 호환용(예: torch==2.4.1+cu124)
ARG SHELL='/bin/bash'
ARG MINICONDA='Miniconda3-py39_23.3.1-0-Linux-x86_64.sh'

ENV LANG=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    CONDA_HOME=/opt/conda \
    SHELL=${SHELL}

ENV PATH=$CONDA_HOME/bin:$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
    LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH \
    CONDA_PREFIX=$CONDA_HOME \
    NCCL_HOME=$CUDA_HOME

# Ubuntu 패키지 설치
RUN apt-get update && apt-get -y install --no-install-recommends \
    python3-pip ffmpeg git less wget libsm6 libxext6 libxrender-dev \
    build-essential cmake pkg-config libx11-dev libatlas-base-dev \
    libgtk-3-dev libboost-python-dev vim libgl1-mesa-glx \
    libaio-dev software-properties-common tmux espeak-ng \
 && rm -rf /var/lib/apt/lists/*

# Miniconda 설치 (Python 3.9)
USER root
RUN wget -t 0 -c -O /tmp/anaconda.sh https://repo.anaconda.com/miniconda/${MINICONDA} \
 && mv /tmp/anaconda.sh /root/anaconda.sh \
 && ${SHELL} /root/anaconda.sh -b -p $CONDA_HOME \
 && rm /root/anaconda.sh

# conda env 생성
RUN conda create -y --name amphion python=3.9.15

WORKDIR /app
COPY env.sh env.sh
RUN chmod +x ./env.sh

# 프로젝트 의존성 설치(사용자 스크립트)
# ⚠️ env.sh 내부에서 torch를 설치한다면, 현재 환경과 맞춰주세요:
#   pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA} torch==${PYTORCH} torchvision torchaudio
RUN ["conda", "run", "-n", "amphion", "-vvv", "--no-capture-output", "./env.sh"]

RUN conda init && echo "\nconda activate amphion\n" >> ~/.bashrc

CMD ["/bin/bash"]

# --- Build/Run 예시 ---
# docker build -t realamphion/amphion .
# docker run --gpus all -it -v $PWD:/app -v /mnt:/mnt_host realamphion/amphion
