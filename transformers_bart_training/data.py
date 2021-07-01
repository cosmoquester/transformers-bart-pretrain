from functools import partial
from typing import Callable, Dict, Tuple

import tensorflow as tf
import tensorflow_text as text


def get_dataset(dataset_file_path: str, tokenizer: text.SentencepieceTokenizer, auto_encoding: bool):
    """
    Read dataset file and construct tensorflow dataset
    :param dataset_file_path:
        - if auto_encoding, text dataset file path just containing text lines.
        - else, tsv dataset file path. formed (sentence1, sentence2) without header.
    :param tokenizer: SentencepieceTokenizer instance.
    :param auto_encoding: whether to use text lines dataset for auto encoding.
                            If true, open dataset files as txt and a lines is an example for auto encoding.
    """

    @tf.function
    def tokenize_fn(source_text, target_text):
        # Tokenize & Add bos, eos
        source_tokens = tokenizer.tokenize(source_text)
        target_tokens = tokenizer.tokenize(target_text)
        return source_tokens, target_tokens

    if auto_encoding:
        duplicate = tf.function(lambda text: (text, text))
        dataset = tf.data.TextLineDataset(
            dataset_file_path,
            num_parallel_reads=tf.data.experimental.AUTOTUNE,
        ).map(duplicate)
    else:
        dataset = tf.data.experimental.CsvDataset(
            dataset_file_path, [tf.string, tf.string], header=True, field_delim="\t"
        )

    return dataset.map(tokenize_fn)


def get_tfrecord_dataset(dataset_file_path: str) -> tf.data.Dataset:
    """Read TFRecord dataset file and construct tensorflow dataset"""

    decompose = tf.function(
        lambda serialized_example: (
            tf.io.parse_tensor(serialized_example[0], tf.int32),
            tf.io.parse_tensor(serialized_example[1], tf.int32),
        )
    )
    dataset = (
        tf.data.TFRecordDataset(dataset_file_path, "GZIP")
        .map(partial(tf.io.parse_tensor, out_type=tf.string))
        .map(decompose)
    )
    return dataset


@tf.function
def make_train_examples(source_tokens: tf.Tensor, target_tokens: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Make training examples from source and target tokens."""
    return {"input_ids": source_tokens, "decoder_input_ids": target_tokens[:-1]}, target_tokens[1:]


def text_infilling(mask_token_id: int):
    mask_token = tf.constant([mask_token_id], tf.int32)

    @tf.function
    def _text_infilling(inputs: Dict[str, tf.Tensor], target: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """Add text infilling noise to example"""
        source_tokens = inputs["input_ids"]
        token_length = tf.shape(source_tokens)[0]
        span_length = tf.minimum(tf.random.poisson((), lam=3, dtype=tf.int32), token_length - 1)
        start_index = tf.random.uniform((), 0, token_length - span_length, tf.int32)
        source_tokens = tf.concat(
            [
                source_tokens[:start_index],
                mask_token,
                source_tokens[start_index + span_length :],
            ],
            axis=0,
        )

        return {"input_ids": source_tokens, "decoder_input_ids": inputs["decoder_input_ids"]}, target

    return _text_infilling


def filter_example(max_sequence_length: int) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    @tf.function
    def _filter(source_tokens: tf.Tensor, target_tokens: tf.Tensor) -> tf.Tensor:
        return tf.math.logical_and(
            tf.size(source_tokens) < max_sequence_length,
            tf.size(target_tokens) < max_sequence_length,
        )

    return _filter


def slice_example(max_sequence_length: int) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    @tf.function
    def _slice(source_tokens: tf.Tensor, target_tokens: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return (
            source_tokens[:max_sequence_length],
            target_tokens[:max_sequence_length],
        )

    return _slice
