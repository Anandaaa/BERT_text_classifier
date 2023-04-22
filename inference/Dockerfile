FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y apt-utils vim

RUN pip install datasets transformers fastapi uvicorn

COPY inference/config.json /workspace/inference/config.json
COPY inference/inference.py /workspace/inference/inference.py
COPY inference/fa_server.py /workspace/inference/fa_server.py
COPY inference/fa_serve.sh /workspace/inference/fa_serve.sh

COPY models/distilbert_finetuned_v3/ /workspace/models/distilbert_finetuned_v3/

WORKDIR /workspace/inference
ENV PATH="/workspace/inference:${PATH}" PYTHONPATH="/workspace/inference:${PYTHONPATH}"