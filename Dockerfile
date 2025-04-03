# Use an NVIDIA CUDA image with cuDNN, for example CUDA 12.4.1 with Ubuntu 24.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

# Set DEBIAN_FRONTEND to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages, including time and wget 
RUN apt-get update && \
    apt-get install -y time wget bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy the updated env.yaml into the container and create the conda environment
COPY env.yaml /tmp/env.yaml
RUN conda env create -f /tmp/env.yaml && conda clean -a

# Set the shell to use the newly created conda environment (replace 'env' with environment name if different)
SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

# Set required environment variables for S3 endpoints if needed.
ENV ENDPOINT_URL="https://s3.braingeneers.gi.ucsc.edu"

# Copy application code into the container and set the working directory.
COPY . /app
WORKDIR /app

# Set the default command. We include stdbuf and /usr/bin/time -v for resource tracking.
CMD ["stdbuf", "-i0", "-o0", "-e0", "/usr/bin/time", "-v", "python", "sorter.py"]
