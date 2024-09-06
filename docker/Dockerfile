FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV DEBIAN_FRONTEN=noninteractive
ENV PYTHONUNBUFFERED=1

# ==================================================================
# Timezone
# ------------------------------------------------------------------
ENV TZ="Asia/Seoul"
ENV LC_ALL="C.UTF-8"
RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul apt-get install -y tzdata
# ------------------------------------------------------------------
# ~Timezone
# ==================================================================

RUN apt-get install -y git vim build-essential curl wget ffmpeg libsndfile1-dev libsox-dev  && \
    apt-get clean

RUN apt install -y python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install "numpy<2.0"

# ==================================================================
# Devel environment
# ------------------------------------------------------------------
RUN wget https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -O ~/.git-completion.bash && \
    echo "source ~/.git-completion.bash" >> ~/.bashrc
# ------------------------------------------------------------------
# ~Devel environment
# ==================================================================


ARG PYTORCH='2.1.0'
ARG TORCH_VISION='0.16.0'
ARG TORCH_AUDIO='2.1.0'
ARG CUDA='cu121'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_VISION} -gt 0 ] && VERSION='torchvision=='$TORCH_VISION ||  VERSION='torchvision'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN [ ${#TORCH_AUDIO} -gt 0 ] && VERSION='torchaudio=='$TORCH_AUDIO ||  VERSION='torchaudio'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA


# ==================================================================
WORKDIR /workspace
RUN git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    pip install -e . && \
    pip install accelerate datasets

RUN git clone https://github.com/openai/whisper.git && \
    cd whisper && \
    pip install -e .

# RUN git clone https://github.com/cpm0722/transformer_pytorch.git
RUN git clone https://github.com/dongwon00kim/transformer_pytorch.git


# ==================================================================

RUN  git clone https://github.com/dongwon00kim/StudyMate-Transformer.git

WORKDIR /workspace/StudyMate-Transformer

ONBUILD RUN git pull