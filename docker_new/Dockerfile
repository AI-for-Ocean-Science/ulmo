FROM continuumio/miniconda3

# get the nenya source code from github; patch with workaround symlink for ulmo.ssl to ulmo.nenya; 
# create additional directories that we'll need.
RUN git clone --depth 1 --branch nenya_resubmit https://github.com/AI-for-Ocean-Science/ulmo.git &&\
    mkdir /ulmo/ulmo/ssl &&\
    ln -s /ulmo/ulmo/nenya/train_util.py /ulmo/ulmo/ssl/train_util.py &&\
    mkdir /nenya_docker &&\
    mkdir /nenya_data

# copy dependencies requirement files
COPY ["environment.yaml", "requirements.txt", "requirements_extra.txt", "/nenya_docker/"]
# copy pytorch file
COPY ["last.pth", "/ulmo/ulmo/nenya/"]

# update conda and create environment
WORKDIR /nenya_docker
RUN conda update --name base conda &&\
    conda env create --file environment.yaml

# install nenya; needs to be done from the newly created conda nenya environment
WORKDIR /ulmo
RUN ["conda", "run", "--no-capture-output", "-n", "nenya", "pip", "install", "-e", ".[dev]"] 

# set nenya environment variables
ENV OS_SST=/ulmo_data
ENV BOKEH_ALLOW_WS_ORIGIN=10.211.55.4:5006
ENV ENDPOINT_URL=https://s3-west.nrp-nautilus.io

# the nenya bokeh service listens on this port
EXPOSE 5006

# switch to nenya directory at start up
WORKDIR /ulmo/ulmo/nenya

# # start nenya with an image test
# CMD ["conda", "run", "--no-capture-output", "-n", "nenya", "nenya_portal", "--image_file", "test_image.npy"]

# start nenya with a table test
CMD ["conda", "run", "--no-capture-output", "-n", "nenya", "python3", "portal.py"] 