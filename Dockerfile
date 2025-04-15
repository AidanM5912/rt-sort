# Use an NVIDIA CUDA image with cuDNN
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set DEBIAN_FRONTEND to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Disable Intel JIT profiling (avoids iJIT_NotifyEvent crash)
ENV IJIT_DISABLE=1

# Install system packages and the NVIDIA package repository for TensorRT
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        time \
        wget \
        bzip2 \
        libnvinfer8 \
        libnvinfer-plugin8 \
        libnvparsers8 \
        libnvonnxparsers8 && \
    rm -rf /var/lib/apt/lists/*

#libnvinfer-dev \
#libnvinfer-plugin-dev \

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy the updated mini.yaml into the container and create the conda environment
COPY mini.yaml /tmp/mini.yaml 
RUN conda env create -f /tmp/mini.yaml && conda clean -a

#stupid pip dependency
#stupid pytorch dependency

RUN conda run -n rt-sort-minimal pip install --upgrade pip==23.3.1 && \
    conda run -n rt-sort-minimal pip install torch-tensorrt==1.2.0 --find-links https://github.com/pytorch/TensorRT/releases/expanded_assets/v1.2.0

#numpy is being upgraded to 1.24.4 which is not what we want, fix
RUN conda run -n rt-sort-minimal pip uninstall -y numpy && \
    conda run -n rt-sort-minimal pip install numpy==1.22.4

# Create the expected HDF5 plugin directory and set the environment variable
RUN mkdir -p /opt/conda/envs/rt-sort-minimal/lib/hdf5/plugin
ENV HDF5_PLUGIN_PATH=/opt/conda/envs/rt-sort-minimal/lib/hdf5/plugin

# Set required environment variables for S3 endpoints if needed.
ENV ENDPOINT_URL="https://s3.braingeneers.gi.ucsc.edu"

# Copy application code into the container and set the working directory.
COPY . /app
WORKDIR /app

#point to conda env for runtime 
ENV LD_LIBRARY_PATH=/opt/conda/envs/rt-sort-minimal/lib:$LD_LIBRARY_PATH

# Copy the proprietary HDF5 compression plugin into the container
COPY libcompression.so /opt/conda/envs/rt-sort-minimal/lib/hdf5/plugin/libcompression.so

# Export the plugin path for HDF5 to find it
ENV HDF5_PLUGIN_PATH=/opt/conda/envs/rt-sort-minimal/lib/hdf5/plugin


# Set the default command. include stdbuf and /usr/bin/time -v for resource tracking.
#CMD ["stdbuf", "-i0", "-o0", "-e0", "/usr/bin/time", "-v", "python", "sorter.py"] # old one doesnt really matter k8s overwrites with command in job yaml but whatever
CMD ["conda", "run", "--no-capture-output", "-n", "rt-sort-minimal", "stdbuf", "-i0", "-o0", "-e0", "/usr/bin/time", "-v", "python", "sorter.py"]
