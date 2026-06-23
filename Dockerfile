# An Image development environment
#     docker build -t cio_image_lab:latest --build-arg user=USERNAME --build-arg group=GROUPNAME --build-arg user_id=USERID --build-arg group_id=GROUPID .
FROM ubuntu:24.04

# Install essential packages
RUN apt-get update \
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND='noninteractive' apt-get install -y \
               python3-pip \
               python3-venv \
               python3-dev \
               nano \
               wget \
               git \
               build-essential \
               sudo \
               libhdf5-dev \
    && apt-get autoremove -y \
    && apt-get clean -y

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages inside the virtual environment
RUN pip install --no-cache-dir --upgrade pip==26.0.1 setuptools==82.0.0 \
    && pip install --no-cache-dir \
                pandas==2.3.3 \
                numpy==1.26.4 \
                scipy \
                h5py==3.15.1 \
                scikit-learn \
                openpyxl \
                umap-learn \
                tables \
                imageio \
                xmltodict \
                scikit-image \
                imagecodecs \
                jsonschema \
                opencv-python-headless \
                pythologist-test-images \
                pyarrow \
                jupyterlab \
                matplotlib \
                plotnine[all] \
                seaborn \
                zarr \
                ome-zarr \
                dask

# Create a user with specific user_id and group_id
ARG user=jupyter_user
ARG user_id=9999
ARG group=jupyter_group
ARG group_id=9999

RUN groupadd -g $group_id $group \
    && useradd -l -u $user_id -ms /bin/bash -g $group $user \
    && usermod -a -G $group $user

# Clone, build, and install custom packages
RUN mkdir /source \
    && git clone https://github.com/jason-weirather/good-neighbors.git /source/good-neighbors \
    && pip install --no-cache-dir -e /source/good-neighbors

ADD . /source/pythologist
RUN pip install --no-cache-dir /source/pythologist

# Create necessary directories with appropriate permissions
RUN mkdir -p /home/$user/.local \
    && mkdir -p /home/$user/.jupyter \
    && mkdir -p /work \
    && chown -R $user:$group /home/$user/.local /home/$user/.jupyter /work

# Create necessary directories with appropriate permissions
RUN mkdir -p /.local /.jupyter /.cache \
    && chmod -R 777 /.local /.jupyter /.cache

# Switch to the new user
USER $user

# Set the working directory
WORKDIR /work

# Command to start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]