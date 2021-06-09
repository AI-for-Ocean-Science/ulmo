FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}" TZ="US"

COPY ./modis_SimCLR /modis_simclr

RUN set -x; buildDeps='wget' \
    && apt-get update \
    && apt-get install -y $buildDeps \
    && apt-get install -y tzdata \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.9.2-Linux-x86_64.sh -b \
    && rm -rf Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && apt-get install -y awscli \
    && apt-get purge -y --auto-remove $buildDeps

RUN pip install --upgrade pip \
    && pip install torch \
    && pip install torchvision \
    && pip install scikit-image \
    && pip install tensorboard_logger \
    && pip install h5py 
   
