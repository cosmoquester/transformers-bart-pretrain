import argparse
import sys
from math import ceil

import tensorflow as tf
from transformers import AdamWeightDecay, BartConfig, TFBartForConditionalGeneration

from transformers_bart_pretrain.data import (
    filter_example,
    get_dataset,
    get_tfrecord_dataset,
    make_train_examples,
    sentence_permutation,
    slice_example,
    text_infilling,
)
from transformers_bart_pretrain.measure import SparseCategoricalAccuracy, SparseCategoricalCrossentropy
from transformers_bart_pretrain.utils import (
    LRScheduler,
    get_device_strategy,
    get_logger,
    path_join,
    set_mixed_precision,
    set_random_seed,
)

# fmt: off
parser = argparse.ArgumentParser("This is script to train seq2seq model")
group = parser.add_argument_group("File Paths")
group.add_argument("--model-config-path", type=str, required=True, help="model config file")
group.add_argument("--train-dataset-path", required=True, help="training dataset, a text file or multiple files ex) *.txt")
group.add_argument("--dev-dataset-path", required=True, help="dev dataset, a text file or multiple files ex) *.txt")
group.add_argument("--pretrained-checkpoint", type=str, default=None, help="pretrained checkpoint path")
group.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
group.add_argument("--sp-model-path", type=str, help="sentencepiece model path to tokenizer")

group = parser.add_argument_group("Training Parameters")
group.add_argument("--mask-token", type=str, help="mask token ex) [MASK]")
group.add_argument("--mask-token-id", type=int, help="mask token id of vocab")
group.add_argument("--epochs", type=int, default=10)
group.add_argument("--steps-per-epoch", type=int, default=None)
group.add_argument("--learning-rate", type=float, default=2e-4)
group.add_argument("--min-learning-rate", type=float, default=1e-5)
group.add_argument("--warmup-steps", type=int)
group.add_argument("--warmup-rate", type=float, default=0.06)
group.add_argument("--batch-size", type=int, default=512, help="total training batch size of all devices")
group.add_argument("--dev-batch-size", type=int, default=512)
group.add_argument("--num-total-dataset", type=int, default=1000000)
group.add_argument("--shuffle-buffer-size", type=int, default=20000)
group.add_argument("--prefetch-buffer-size", type=int, default=1000)
group.add_argument("--max-sequence-length", type=int, default=256)
group.add_argument("--weight-decay", type=float, default=0.0, help="use weight decay")
group.add_argument("--clipnorm", type=float, help="clips gradients to a maximum norm.")
group.add_argument("--disable-text-infilling", action="store_false", dest="use_text_infilling", help="disable input noising")
group.add_argument("--disable-sentence-permutation", action="store_false", dest="use_sentence_permutation", help="disable input noising")
group.add_argument("--masking-rate", type=float, default=0.3, help="text infilling masking rate")
group.add_argument("--permutation-segment-token-id", type=int, required=True, help="segment token id for sentence permutation")

group = parser.add_argument_group("Other settings")
group.add_argument("--tensorboard-update-freq", type=int, help='log losses and metrics every after this value step')
group.add_argument("--mixed-precision", action="store_true", help="Use mixed precision FP16")
group.add_argument("--auto-encoding", action="store_true", help="train by auto encoding with text lines dataset")
group.add_argument("--use-tfrecord", action="store_true", help="train using tfrecord dataset")
group.add_argument("--repeat-each-file", action="store_true", dest="repeat", help="repeat each dataset and uniform sample for train example")
group.add_argument("--debug-nan-loss", action="store_true", help="Trainin with this flag, print the number of Nan loss (not supported on TPU)")
group.add_argument("--seed", type=int, help="random seed")
group.add_argument("--skip-epochs", type=int, default=0, help="skip this number of epochs")
group.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU", "TPU"], help="device to train model")
group.add_argument("--max-over-sequence-policy", type=str, default="slice", choices=["filter", "slice"], help="Policy for sequences of which length is over the max")
# fmt: on


def main(args: argparse.Namespace):
    strategy = get_device_strategy(args.device)

    logger = get_logger(__name__)

    if args.mixed_precision:
        logger.info("[+] Use Mixed Precision FP16")
        set_mixed_precision(args.device)

    if args.seed:
        logger.info(f"[+] Set random seed to {args.seed}")
        set_random_seed(args.seed)

    # Copy config file
    tf.io.gfile.makedirs(args.output_path)
    with tf.io.gfile.GFile(path_join(args.output_path, "argument_configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")
    tf.io.gfile.copy(args.model_config_path, path_join(args.output_path, "model_config.yml"))

    train_dataset_files = tf.io.gfile.glob(args.train_dataset_path)
    dev_dataset_files = tf.io.gfile.glob(args.dev_dataset_path)
    if not train_dataset_files or not dev_dataset_files:
        raise RuntimeError("Dataset path is invalid!")

    if args.sp_model_path:
        import tensorflow_text as text

        logger.info("[+] Load Tokenizer")
        with tf.io.gfile.GFile(args.sp_model_path, "rb") as f:
            tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)
    else:
        logger.warn("[-] Tokenizer path not passed, Exception may be raised!")

    if args.mask_token_id:
        mask_token_id = args.mask_token_id
    elif args.mask_token:
        mask_token_id = tokenizer.string_to_id(args.mask_token).numpy()
    else:
        raise RuntimeError("You should set `--mask-token-id` or `--mask-token`")

    with strategy.scope():
        logger.info("[+] Load Dataset")
        if args.use_tfrecord:
            train_dataset = get_tfrecord_dataset(args.train_dataset_path, args.repeat)
            dev_dataset = get_tfrecord_dataset(args.dev_dataset_path, False)
        else:
            train_dataset = get_dataset(args.train_dataset_path, tokenizer, args.auto_encoding, args.repeat)
            dev_dataset = get_dataset(args.dev_dataset_path, tokenizer, args.auto_encoding, False)

        # Apply policy for sequences whose length is over than max sequence length
        if args.max_over_sequence_policy == "filter":
            logger.info(f"[+] Filter examples whose sequence length is over than {args.max_sequence_length}")
            train_dataset = train_dataset.filter(filter_example(args.max_sequence_length))
            dev_dataset = dev_dataset.filter(filter_example(args.max_sequence_length))
        elif args.max_over_sequence_policy == "slice":
            logger.info(f"[+] Slice examples whose sequence length is over than {args.max_sequence_length}")
            train_dataset = train_dataset.map(
                slice_example(args.max_sequence_length), num_parallel_calls=tf.data.AUTOTUNE
            )
            dev_dataset = dev_dataset.map(slice_example(args.max_sequence_length), num_parallel_calls=tf.data.AUTOTUNE)

        if args.steps_per_epoch:
            logger.info("[+] Repeat dataset")
            train_dataset = train_dataset.repeat()

            if args.skip_epochs:
                logger.info(
                    f"[+] Skip examples by {args.skip_epochs}epoch x {args.steps_per_epoch} steps x {args.batch_size} batches"
                )
                train_dataset = train_dataset.skip(args.skip_epochs * args.steps_per_epoch * args.batch_size)
        elif not args.num_total_dataset:
            raise RuntimeError("You should pass `--num-total-dataset` or `--steps-per-epoch` for lr scheduling!")
        elif args.repeat:
            raise RuntimeError("You should pass `--steps-per-epoch` when using `--repeat-each-file`!")

        # Make into Training Examples
        train_dataset = train_dataset.shuffle(args.shuffle_buffer_size).map(
            make_train_examples, num_parallel_calls=tf.data.AUTOTUNE
        )
        dev_dataset = dev_dataset.map(make_train_examples)

        # Apply Noise Functions
        if args.use_text_infilling:
            logger.info("[+] Apply Text Infilling Noise")
            text_infilling_fn = text_infilling(mask_token_id, args.masking_rate)
            train_dataset = train_dataset.map(text_infilling_fn, tf.data.AUTOTUNE)
            dev_dataset = dev_dataset.map(text_infilling_fn, tf.data.AUTOTUNE)

        if args.use_sentence_permutation:
            logger.info("[+] Apply Sentence Permutation Noise")
            sentence_permutation_fn = sentence_permutation(args.permutation_segment_token_id)
            train_dataset = train_dataset.map(sentence_permutation_fn, tf.data.AUTOTUNE)
            dev_dataset = dev_dataset.map(sentence_permutation_fn, tf.data.AUTOTUNE)

        logger.info("[+] Initialize Model")
        model_config = BartConfig.from_pretrained(args.model_config_path)
        model = TFBartForConditionalGeneration(model_config)

        # Batching
        pad_shape = (
            {
                "input_ids": [args.max_sequence_length + 1],
                "attention_mask": [args.max_sequence_length + 1],
                "decoder_input_ids": [args.max_sequence_length - 1],
            },
            [args.max_sequence_length - 1],
        )
        pad_values = (
            {
                "input_ids": model_config.pad_token_id,
                "attention_mask": 0,
                "decoder_input_ids": model_config.pad_token_id,
            },
            model_config.pad_token_id,
        )
        train_dataset = train_dataset.padded_batch(args.batch_size, pad_shape, pad_values).prefetch(
            args.prefetch_buffer_size
        )
        dev_dataset = dev_dataset.padded_batch(args.dev_batch_size, pad_shape, pad_values)

        model(
            {
                "input_ids": tf.keras.Input([args.max_sequence_length + 1], dtype=tf.int32),
                "attention_mask": tf.keras.Input([args.max_sequence_length + 1], dtype=tf.int32),
                "decoder_input_ids": tf.keras.Input([args.max_sequence_length - 1], dtype=tf.int32),
            }
        )
        model.summary()

        if args.pretrained_checkpoint:
            logger.info("[+] Load weights of trained model")
            model.load_weights(args.pretrained_checkpoint)

        logger.info("[+] Compile Model")
        total_steps = (args.steps_per_epoch or ceil(args.num_total_dataset / args.batch_size)) * args.epochs
        offset_steps = (args.steps_per_epoch or ceil(args.num_total_dataset / args.batch_size)) * args.skip_epochs
        learning_rate = LRScheduler(
            total_steps, args.learning_rate, args.min_learning_rate, args.warmup_rate, args.warmup_steps, offset_steps
        )

        logger.info("[+] Use AdamW Optimizer")
        optimizer = AdamWeightDecay(
            learning_rate,
            weight_decay_rate=args.weight_decay,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            clipnorm=args.clipnorm,
        )

        model.compile(
            optimizer=optimizer,
            loss={
                "logits": SparseCategoricalCrossentropy(model_config.pad_token_id, from_logits=True),
                "encoder_last_hidden_state": None,
            },
            metrics={"logits": SparseCategoricalAccuracy(model_config.pad_token_id)},
        )

        logger.info("[+] Start training")
        model.fit(
            train_dataset,
            validation_data=dev_dataset,
            initial_epoch=args.skip_epochs,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    path_join(
                        args.output_path,
                        "models",
                        "model-{epoch}epoch-{val_loss:.4f}loss_{val_logits_accuracy:.4f}acc.ckpt",
                    ),
                    save_weights_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.TensorBoard(
                    path_join(args.output_path, "logs"),
                    update_freq=args.tensorboard_update_freq if args.tensorboard_update_freq else "batch",
                ),
            ],
        )
        logger.info("[+] Finished training!")


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
