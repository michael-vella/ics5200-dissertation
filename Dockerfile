FROM python:3.12-slim

# cpu -> requirements-cpu.txt, gpu -> requirements-gpu.txt (CUDA runtime ships inside the torch wheel)
ARG VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

# RDKit drawing needs X render libs; git for any VCS-based installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-cpu.txt requirements-gpu.txt ./
RUN pip install --upgrade pip && pip install -r requirements-${VARIANT}.txt

COPY . .

RUN mkdir -p /cache/huggingface data/processed experiments/entries \
    && chmod -R a+rwX /cache /app

CMD ["bash"]
