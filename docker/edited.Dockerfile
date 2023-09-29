FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ARG USER_NAME=zed
ARG USER_ID=1000


# Prevent anything requiring user input
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

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
    && rm -rf /var/lib/apt/lists/*

# Update torch and torchvision
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV
# RUN pip install mmcv-full==1.2.7+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
RUN pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6/index.html
# RUN pip install mmdet==2.10.0

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/saic-vul/imvoxelnet.git /mmdetection3d
WORKDIR /mmdetection3d
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
# RUN pip uninstall pycocotools --no-cache-dir -y
# RUN pip install mmpycocotools==12.0.3 --no-cache-dir --force --no-deps

# Install differentiable IoU
RUN git clone https://github.com/lilanxiao/Rotated_IoU /rotated_iou
RUN cp -r /rotated_iou/cuda_op /mmdetection3d/mmdet3d/ops/rotated_iou
WORKDIR /mmdetection3d/mmdet3d/ops/rotated_iou/cuda_op
RUN python setup.py install

RUN useradd -m -l -u ${USER_ID} -s /bin/bash ${USER_NAME} \
    && usermod -aG video ${USER_NAME}

# Give them passwordless sudo
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to user to run user-space commands
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

USER root
# entrypoint declaration.
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT [ "./entrypoint.sh" ]