import argparse
import json
import sys

import tensorflow as tf

from sample_package.data import get_dataset
from sample_package.model import SampleModel
from sample_package.utils import get_device_strategy, get_logger, path_join

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--model-config-path", type=str, required=True, help="model config file")
parser.add_argument("--dataset-path", required=True, help="a text file or multiple files ex) *.txt")
parser.add_argument("--pretrained-model-path", type=str, default=None, help="pretrained model checkpoint")
parser.add_argument("--shuffle-buffer-size", type=int, default=5000)
parser.add_argument("--output-path", default="output", help="output directory to save log and model checkpoints")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--steps-per-valid", type=int, default=1000)
parser.add_argument("--steps-per-save", type=int, default=2000)
parser.add_argument("--learning-rate", type=float, default=2e-3)
parser.add_argument("--min-learning-rate", type=float, default=1e-8)
parser.add_argument("--total-data-size", type=int, help="the number of training dataset examples")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--dev-batch-size", type=int, default=2)
parser.add_argument("--num-dev-dataset", type=int, default=2)
parser.add_argument("--tensorboard-update-freq", type=int, default=1)
parser.add_argument("--disable-mixed-precision", action="store_false", dest="mixed_precision", help="Use mixed precision FP16")
parser.add_argument("--device", type=str, default="CPU", help="device to use (TPU or GPU or CPU)")
# fmt: on

if __name__ == "__main__":
    args = parser.parse_args()

    logger = get_logger()

    if args.mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logger.info("Use Mixed Precision FP16")

    # Copy config file
    tf.io.gfile.makedirs(args.output_path)
    with tf.io.gfile.GFile(path_join(args.output_path, "argument_configs.txt"), "w") as fout:
        for k, v in vars(args).items():
            fout.write(f"{k}: {v}\n")
    tf.io.gfile.copy(args.model_config_path, path_join(args.output_path, "model_config.json"))

    strategy = get_device_strategy(args.device)
    with strategy.scope():
        # Construct Dataset
        dataset_files = tf.io.gfile.glob(args.dataset_path)
        if not dataset_files:
            logger.error("[Error] Dataset path is invalid!")
            sys.exit(1)

        dataset = get_dataset(dataset_files).shuffle(args.shuffle_buffer_size)
        train_dataset = dataset.skip(args.num_dev_dataset).batch(args.batch_size)
        dev_dataset = dataset.take(args.num_dev_dataset).batch(max(args.batch_size, args.dev_batch_size))

        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        dev_dataset = strategy.experimental_distribute_dataset(dev_dataset)

        # Model Initialize
        with tf.io.gfile.GFile(args.model_config_path) as f:
            model = SampleModel(**json.load(f))

        # Load pretrained model
        if args.pretrained_model_path:
            model.load_weights(args.pretrained_model_path)
            logger.info("Loaded weights of model")

        # Set criterion & optimizer
        criterion = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        if args.total_data_size:
            total_step = (args.total_data_size - args.num_dev_dataset) // args.batch_size
            learning_rate = learning_rate_scheduler(
                args.steps_per_epoch * args.epochs, args.learning_rate, args.min_learning_rate
            )
        else:
            total_step = None
            learning_rate = args.learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        logger.info("Set optimizer and loss")
        logger.info("Start training")

        # Metrics
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        valid_loss = tf.keras.metrics.Mean(name="valid_loss")
        train_acc = tf.keras.metrics.BinaryAccuracy(name="train_acc")
        valid_acc = tf.keras.metrics.BinaryAccuracy(name="valid_acc")

        # Training
        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                features, labels = inputs
                global_batch_size = tf.shape(labels)[0] * strategy.num_replicas_in_sync

                with tf.GradientTape() as tape:
                    preds = model(features, training=True)
                    dist_loss = criterion(labels, preds)
                    loss = tf.reduce_sum(dist_loss) / tf.cast(global_batch_size, dist_loss.dtype)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
                return loss, (labels, preds)

            loss, (labels, preds) = strategy.run(step_fn, args=(dist_inputs,))
            train_loss.update_state(loss)
            train_acc.update_state(labels, preds)
            return loss

        @tf.function
        def valid_step(dist_inputs):
            def step_fn(inputs):
                features, labels = inputs
                global_batch_size = tf.shape(labels)[0] * strategy.num_replicas_in_sync

                preds = model(features)
                dist_loss = criterion(labels, preds)
                loss = tf.reduce_sum(dist_loss) / tf.cast(global_batch_size, dist_loss.dtype)
                return loss, (labels, preds)

            loss, (labels, preds) = strategy.run(step_fn, args=(dist_inputs,))
            valid_loss.update_state(loss)
            valid_acc.update_state(labels, preds)
            return loss

        for epoch in range(1, args.epochs + 1):
            progress_bar = tf.keras.utils.Progbar(total_step)
            logger.info(f"Train {epoch} epoch")
            for step, dist_inputs in enumerate(train_dataset, start=1):
                train_step(dist_inputs)
                progress_bar.update(
                    step,
                    values=[
                        (train_loss.name, train_loss.result()),
                        (train_acc.name, train_acc.result()),
                    ],
                )

                # Validation
                if step % args.steps_per_valid == 0 or step == total_step:
                    for dist_val_inputs in dev_dataset:
                        valid_step(dist_val_inputs)

                    print()
                    logger.info(f"{epoch} epoch {step} / {total_step} steps, valid_loss: {valid_loss.result()}")
                    valid_loss.reset_states()

                # TODO: Tensorboard
                if step % args.tensorboard_update_freq == 0:
                    pass

                # Save Checkpoint
                if step % args.steps_per_save == 0 or step == total_step:
                    model_save_path = path_join(
                        args.output_path,
                        "models",
                        f"model-{epoch}epoch-{valid_loss.result():.4f}loss_{valid_acc.result():.4f}acc.ckpt",
                    )
                    model.save_weights(model_save_path)

            train_loss.reset_states()
            train_acc.reset_states()

            # Set total step for next batch
            if total_step is None:
                total_step = step
