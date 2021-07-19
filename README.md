# transformers TF BART pre-training

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![cosmoquester](https://circleci.com/gh/cosmoquester/transformers-bart-training.svg?style=svg)](https://app.circleci.com/pipelines/github/cosmoquester/transformers-bart-training)
[![codecov](https://codecov.io/gh/cosmoquester/transformers-bart-training/branch/master/graph/badge.svg?token=FT7NreB8Ku)](https://codecov.io/gh/cosmoquester/transformers-bart-training)

- Script to pre-train hugginface transformers BART
- Training [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- `Text infilling` is the only noise function available now

# Train

You can train huggingface transformers model simply like below example.
(below example works without change as itself using sample data)

```sh
$ CUDA_VISIBLE_DEVICES=1 python -m scripts.train \
    --model-config-path configs/base.json \
    --train-dataset-path tests/data/sample1.txt \
    --dev-dataset-path tests/data/sample1.txt \
    --sp-model-path sp_model/sp_model_unigram_8K.model \
    --device GPU \
    --auto-encoding \
    --batch-size 2 \
    --steps-per-epoch 100 \
    --mask-token "[MASK]" \
    --mixed-precision
```

## Arguments

```sh
File Paths:
  --model-config-path MODEL_CONFIG_PATH
                        model config file
  --train-dataset-path TRAIN_DATASET_PATH
                        training dataset, a text file or multiple files ex)
                        *.txt
  --dev-dataset-path DEV_DATASET_PATH
                        dev dataset, a text file or multiple files ex) *.txt
  --pretrained-checkpoint PRETRAINED_CHECKPOINT
                        pretrained checkpoint path
  --output-path OUTPUT_PATH
                        output directory to save log and model checkpoints
  --sp-model-path SP_MODEL_PATH

Training Parameters:
  --mask-token MASK_TOKEN
                        mask token ex) [MASK]
  --mask-token-id MASK_TOKEN_ID
                        mask token id of vocab
  --epochs EPOCHS
  --steps-per-epoch STEPS_PER_EPOCH
  --learning-rate LEARNING_RATE
  --min-learning-rate MIN_LEARNING_RATE
  --warmup-steps WARMUP_STEPS
  --warmup-rate WARMUP_RATE
  --batch-size BATCH_SIZE
                        total training batch size of all devices
  --dev-batch-size DEV_BATCH_SIZE
  --num-total-dataset NUM_TOTAL_DATASET
  --shuffle-buffer-size SHUFFLE_BUFFER_SIZE
  --prefetch-buffer-size PREFETCH_BUFFER_SIZE
  --max-sequence-length MAX_SEQUENCE_LENGTH
  --weight-decay WEIGHT_DECAY
                        use weight decay

Other settings:
  --tensorboard-update-freq TENSORBOARD_UPDATE_FREQ
                        log losses and metrics every after this value step
  --mixed-precision     Use mixed precision FP16
  --auto-encoding       train by auto encoding with text lines dataset
  --use-tfrecord        train using tfrecord dataset
  --repeat-each-file    repeat each dataset and uniform sample for train
                        example
  --debug-nan-loss      Trainin with this flag, print the number of Nan loss
                        (not supported on TPU)
  --seed SEED           random seed
  --skip-epochs SKIP_EPOCHS
                        skip this number of epochs
  --device {CPU,GPU,TPU}
                        device to train model
  --max-over-sequence-policy {filter,slice}
                        Policy for sequences of which length is over the max
```
- `model-config-path` is huggingface bart model config file path.
- `pretrained-checkpoint` is trained model checkpoint path.
- `sp-model-path` is sentencepiece tokenizer model path.
- with `repeat-each-file` flag, you can repeat each dataset files forever even if one of dataset were run out.
