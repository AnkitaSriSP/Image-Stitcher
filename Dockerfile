FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /data

# Install only minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Convenience symlinks (force overwrite if exists)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    numpy==1.24.4 \
    pillow==11.2.1 \
    tqdm==4.67.1

# Pre-download ResNet18 weights
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"

# Copy your script into the working directory
COPY . /data

# Set entrypoint to run the script
ENTRYPOINT ["python", "main.py"]
