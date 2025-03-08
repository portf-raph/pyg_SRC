FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu20.04
# set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# install system dependencies
ARG PYTHON_VERSION=3.9

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
	  git \
          python${PYTHON_VERSION} \
          python3-pip \
          python${PYTHON_VERSION}-dev \
# change default python
    && cd /usr/bin \
    && ln -sf python${PYTHON_VERSION}         python3 \
    && ln -sf python${PYTHON_VERSION}m        python3m \
    && ln -sf python${PYTHON_VERSION}-config  python3-config \
    && ln -sf python${PYTHON_VERSION}m-config python3m-config \
    && ln -sf python3                         /usr/bin/python \
# update pip and add common packages
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade \
        setuptools \
        wheel \
        six \
# cleanup
    && apt-get clean \
    && rm -rf $HOME/.cache/pip

# set the working directory
WORKDIR /app
COPY . /app

# requirements | PLEASE NOTE: To use this Dockerfile, requirements.txt must contain exactly
# torchopt, pytorch_geometric, jupyter
RUN python3 -m pip install torch && \
    # use the installed Torch version to build scatter/sparse URLs
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)") && \
    python3 -m pip install \
      torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}.html" && \
    python3 -m pip install \
      torch-sparse -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}.html" && \
    # install remaining requirements
    python3 -m pip install -r requirements.txt

ENV NAME docker-jp
# allow port use
EXPOSE 8888

# run jupyter immediately
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
