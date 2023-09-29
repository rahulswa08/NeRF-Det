# ARG BASE_IMAGE=pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
# ARG BASE_IMAGE=pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
# ARG BASE_IMAGE=pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
# docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
# docker pull pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime



ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:20.12-py3
FROM ${BASE_IMAGE}

ARG USER_NAME=zed
ARG USER_ID=1000


# Prevent anything requiring user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.6"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"



# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Basic packages
RUN apt-get -y update \
    && apt-get -y install \
      python3-pip \
      sudo \
      vim \
      wget \
      curl \
      software-properties-common \
      doxygen \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update && apt-get -y install \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6\
    libxrender-dev \
    libxext6 \
    libnccl2=2.8.3-1+cuda11.1 \ 
    libnccl-dev=2.8.3-1+cuda11.1 \
    && rm -rf /var/lib/apt/lists/*
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb

# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# # RUN dpkg -i nccl-repo-2.7.8.deb
# RUN apt-get -y update && apt-get -y install \
#     libnccl2 \
#     libnccl-dev \ 
#     && rm -rf /var/lib/apt/lists/*
# RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# # Install MMCV and MMDetection
# RUN pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# RUN conda install cython
# RUN pip install mmdet==2.10.0

# # Install MMDetection3D (ImVoxelNet)
# RUN conda clean --all
# RUN git clone https://github.com/samsunglabs/imvoxelnet /mmdetection3d
# WORKDIR /mmdetection3d
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .

# # Uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
# RUN pip uninstall pycocotools --no-cache-dir -y
# RUN   

# # Install differentiable IoU
# RUN git clone https://github.com/lilanxiao/Rotated_IoU /rotated_iou
# RUN cp -r /rotated_iou/cuda_op /mmdetection3d/mmdet3d/ops/rotated_iou
# WORKDIR /mmdetection3d/mmdet3d/ops/rotated_iou/cuda_op
# RUN python setup.py install

# RUN pip install torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# RUN pip install mmcv-full==1.3.0 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
# RUN pip install mmdet==2.12.0

# # Install MMDetection
# RUN conda clean --all
# RUN git clone https://github.com/saic-vul/imvoxelnet.git /mmdetection3d
# WORKDIR /mmdetection3d
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .

# # Uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
# RUN pip uninstall pycocotools --no-cache-dir -y
# RUN pip install mmpycocotools==12.0.3 --no-cache-dir --force --no-deps

# # Install differentiable IoU
# RUN git clone https://github.com/lilanxiao/Rotated_IoU /rotated_iou
# RUN cp -r /rotated_iou/cuda_op /mmdetection3d/mmdet3d/ops/rotated_iou
# WORKDIR /mmdetection3d/mmdet3d/ops/rotated_iou/cuda_op
# RUN python setup.py install

RUN useradd -m -l -u ${USER_ID} -s /bin/bash ${USER_NAME} \
    && usermod -aG video ${USER_NAME}

# Give them passwordless sudo
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to user to run user-space commands
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}


# This overrides the default CarlaSim entrypoint, which we want. Theirs starts the simulator.
COPY ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]