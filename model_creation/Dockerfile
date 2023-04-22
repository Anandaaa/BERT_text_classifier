FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN pip install datasets transformers scikit-learn

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y apt-utils vim

COPY finetune_model.py /workspace/finetune_model.py

RUN groupadd -g 1000 ananda && usermod -a -G ananda root