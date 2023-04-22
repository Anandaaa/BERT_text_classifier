THIS_FILES_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run -it \
    -v ${THIS_FILES_DIR}/data:/workspace/data \
    -v ${THIS_FILES_DIR}/models:/workspace/models \
    -v ~/.cache/huggingface/:/workspace/huggingface_cache/ \
    bert_finetuner:v2 ./finetune_model.py \
    -d data/sem_eval_2018_task1 \
    -m models/bert_finetuned_v4 \
    -c huggingface_cache/hub \
    --use_gpu