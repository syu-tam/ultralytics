FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY . /workspace/ultralytics
RUN pip install -e /workspace/ultralytics