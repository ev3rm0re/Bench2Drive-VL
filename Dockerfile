FROM docker.1ms.run/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# ----------------- 基础环境与配置 -----------------
# Retry wrapper
SHELL ["/bin/bash", "-c"]

# Add retry helper
RUN printf '#!/bin/bash\nfor i in {1..6}; do "$@" && break || sleep 10; done\n' > /usr/local/bin/retry && \
    chmod +x /usr/local/bin/retry

ENV DEBIAN_FRONTEND=noninteractive

# 替换 APT 国内镜像源 (阿里云)
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get clean

# Locale setup
RUN retry apt-get update && \
    retry apt-get install --reinstall -y locales && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US \
    LC_ALL=en_US.UTF-8

# ----------------- 系统依赖安装 -----------------
RUN retry apt-get update && \
    retry apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng16-16 \
        libtiff5 \
        libpng-dev \
        python3-dev \
        python3-pip \
        python3-setuptools && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# ----------------- Conda 环境配置 -----------------
ENV CONDA_OVERRIDE_CHANNELS=1 \
    CONDA_ALWAYS_YES=1

# 下载 Miniconda (使用清华镜像)
RUN set -e; \
    RETRIES=5; \
    until curl -o ~/miniconda.sh -LO https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh; do \
        RETRIES=$((RETRIES-1)); \
        if [ $RETRIES -le 0 ]; then echo "Failed to download miniconda.sh"; exit 1; fi; \
        echo "Retry downloading miniconda.sh..."; sleep 3; \
    done && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

ENV PATH="/opt/conda/envs/b2dvl/bin:/opt/conda/envs/bin:/opt/conda/bin:$PATH"

# 配置 Conda 国内镜像源，同意服务条款，并创建环境
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --set show_channel_urls yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    RETRIES=5; \
    until conda create -n b2dvl python=3.7 numpy networkx scipy six requests -y; do \
        RETRIES=$((RETRIES-1)); \
        if [ $RETRIES -le 0 ]; then echo "Failed to create conda environment"; exit 1; fi; \
        echo "Retry creating conda environment..."; sleep 3; \
    done

# ----------------- 项目目录与环境变量 -----------------
WORKDIR /workspace/Bench2Drive-VL
ENV WORKDIR="/workspace/Bench2Drive-VL"

# 调整 CARLA_ROOT 匹配挂载目录名称
ENV CARLA_ROOT="/workspace/CARLA_0.9.15" \
    SCENARIO_RUNNER_ROOT="${WORKDIR}/scenario_runner" \
    LEADERBOARD_ROOT="${WORKDIR}/leaderboard" \
    TEAM_CODE_ROOT="${WORKDIR}/team_code" \
    ADAPTER_ROOT="${WORKDIR}/B2DVL_Adapter" \
    VQA_GEN=1 \
    STRICT_MODE=1

# ----------------- Python 依赖安装 -----------------
# 仅拷贝 requirements 依赖文件以利用 Docker 缓存预装环境
COPY scenario_runner/requirements.txt ${SCENARIO_RUNNER_ROOT}/requirements.txt
COPY leaderboard/requirements.txt ${LEADERBOARD_ROOT}/requirements.txt
COPY requirements.txt /tmp/requirements.txt

# pip 安装依赖
RUN retry pip3 install -r ${SCENARIO_RUNNER_ROOT}/requirements.txt && \
    retry pip3 install -r ${LEADERBOARD_ROOT}/requirements.txt && \
    retry pip3 install -r /tmp/requirements.txt

RUN mkdir -p /workspace/results

# ----------------- PyTorch 与缓存配置 -----------------
# 设置 egg 缓存路径
ENV PYTHON_EGG_CACHE=/root/.egg-cache
RUN mkdir -p $PYTHON_EGG_CACHE

# 安装 PyTorch (精确指定 1.13.1 版本 + CUDA 11.7，并使用阿里云 PyTorch 镜像加速)
RUN bash -c '\
    for i in {1..6}; do \
        pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
        -f https://mirrors.aliyun.com/pytorch-wheels/cu117/ && break || sleep 5; \
    done \
'

# Python path setup (使用 ${PYTHONPATH:-} 修复未定义警告)
ENV PYTHONPATH="${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${TEAM_CODE_ROOT}:${ADAPTER_ROOT}:${PYTHONPATH:-}"

CMD ["/bin/bash"]