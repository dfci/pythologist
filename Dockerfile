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
               r-base \
               build-essential \
               sudo \
               libhdf5-dev \
    && apt-get autoremove -y \
    && apt-get clean -y

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and setuptools inside the virtual environment
RUN pip install --upgrade pip setuptools

# Install Python packages inside the virtual environment
RUN pip install pandas==2.2.0 \
    && pip install numpy==1.26.3 \
    && pip install scipy \
    && pip install h5py==3.15.1 \
    && pip install scikit-learn \
    && pip install openpyxl \
    && pip install umap-learn \
    && pip install tables \
    && pip install imageio \
    && pip install xmltodict \
    && pip install scikit-image \
    && pip install imagecodecs \
    && pip install jsonschema \
    && pip install opencv-python-headless \
    && pip install pythologist-test-images \
    && pip install pyarrow

# Create a user with specific user_id and group_id
ARG user=jupyter_user
ARG user_id=9999
ARG group=jupyter_group
ARG group_id=9999

RUN groupadd -g $group_id $group \
    && useradd -l -u $user_id -ms /bin/bash -g $group $user \
    && usermod -a -G $group $user

# Install additional Python packages inside the virtual environment
RUN pip install jupyterlab \
    && pip install matplotlib \
    && pip install plotnine[all] \
    && pip install seaborn 

# Clone and install the 'good-neighbors' repository inside the virtual environment
RUN mkdir /source \
    && cd /source \
    && git clone https://github.com/jason-weirather/good-neighbors.git \
    && cd good-neighbors \
    && pip install -e .

# Add and install your own package inside the virtual environment
ADD . /source/pythologist
RUN cd /source/pythologist \
    && pip install .

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

