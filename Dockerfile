FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
	python3 \
	python3-pip \
	python3-venv \
	git \
	curl \
	&& rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

RUN pip install \
	torch==2.11.0 \
	--index-url https://download.pytorch.org/whl/cu128

RUN pip install \
	transformers \
	accelerate \
	safetensors \
	peft \
	datasets \
	trl \
	numpy

COPY . /workspace

CMD ["python", "--version"]
