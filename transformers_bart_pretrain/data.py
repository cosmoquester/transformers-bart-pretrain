from functools import partial
from typing import Callable, Dict, Tuple

import tensorflow as tf


def get_dataset(dataset_paths: str, tokenizer, auto_encoding: bool, repeat=False):
    """
    Read dataset file and construct tensorflow dataset.
    Please warn that this function load your multiple dataset files uniformly with `repeat=True`.
    Even if one of the file is run out, repeat that file and use examples from all files uniformly.

    :param dataset_paths:
        - if auto_encoding, text dataset file glob pattern just containing text lines.
        - else, tsv dataset file glob pattern. formed (sentence1, sentence2) without header.
    :param tokenizer: SentencepieceTokenizer instance.
    :param auto_encoding: whether to use text lines dataset for auto encoding.
                            If true, open dataset files as txt and a lines is an example for auto encoding.
    :param repeat: whether repeating each of dataset.
    """
    dataset_list = tf.io.gfile.glob(dataset_paths)
    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)

    @tf.function
    def _tokenize_fn(source_text, target_text) -> Tuple[tf.Tensor, tf.Tensor]:
        # Tokenize & Add bos, eos
        source_tokens = tokenizer.tokenize(source_text)
        target_tokens = tokenizer.tokenize(target_text)
        return source_tokens, target_tokens

    def _to_dataset(dataset_path) -> tf.data.Dataset:
        if auto_encoding:
            duplicate = tf.function(lambda text: (text, text))
            dataset = tf.data.TextLineDataset(
                dataset_path,
                num_parallel_reads=tf.data.AUTOTUNE,
            ).map(duplicate)
        else:
            dataset = tf.data.experimental.CsvDataset(
                dataset_path, [tf.string, tf.string], header=True, field_delim="\t"
            )

        dataset = dataset.map(_tokenize_fn)
        return dataset.repeat() if repeat else dataset

    cycle_length = len(dataset_list) if repeat else None
    return dataset.interleave(_to_dataset, cycle_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_tfrecord_dataset(dataset_paths: str, repeat=False) -> tf.data.Dataset:
    """Read TFRecord dataset file and construct tensorflow dataset"""

    dataset_list = tf.io.gfile.glob(dataset_paths)
    dataset = tf.data.Dataset.from_tensor_slices(dataset_list)

    def _to_dataset(dataset_path) -> tf.data.Dataset:
        decompose = tf.function(
            lambda serialized_example: (
                tf.io.parse_tensor(serialized_example[0], tf.int32),
                tf.io.parse_tensor(serialized_example[1], tf.int32),
            )
        )
        dataset = (
            tf.data.TFRecordDataset(dataset_path, "GZIP")
            .map(partial(tf.io.parse_tensor, out_type=tf.string))
            .map(decompose)
        )
        return dataset.repeat() if repeat else dataset

    cycle_length = len(dataset_list) if repeat else None
    return dataset.interleave(_to_dataset, cycle_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@tf.function
def make_train_examples(source_tokens: tf.Tensor, target_tokens: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Make training examples from source and target tokens."""
    return {
        "input_ids": source_tokens,
        "attention_mask": tf.ones_like(source_tokens, dtype=tf.int32),
        "decoder_input_ids": target_tokens[:-1],
    }, target_tokens[1:]


def text_infilling(mask_token_id: int, masking_rate: float = 0.3):
    mask_token = tf.constant([mask_token_id], tf.int32)

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec(shape=[None], dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=[None], dtype=tf.int32),
                "decoder_input_ids": tf.TensorSpec(shape=[None], dtype=tf.int32),
            },
            tf.TensorSpec(shape=[None], dtype=tf.int32),
        ]
    )
    def _text_infilling(inputs: Dict[str, tf.Tensor], target: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """Add text infilling noise to example"""
        source_tokens = inputs["input_ids"]
        token_length = tf.shape(source_tokens)[0]
        masking_length = tf.cast(tf.cast(token_length, tf.float32) * masking_rate, tf.int32)
        masked_length = 0

        while masked_length < masking_length:
            tf.autograph.experimental.set_loop_options(shape_invariants=[(source_tokens, tf.TensorShape([None]))])
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
            token_length -= span_length - 1
            masked_length += span_length

        return {
            "input_ids": source_tokens,
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": inputs["decoder_input_ids"],
        }, target

    return _text_infilling


def sentence_permutation(segment_token_id: int):
    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec(shape=[None], dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=[None], dtype=tf.int32),
                "decoder_input_ids": tf.TensorSpec(shape=[None], dtype=tf.int32),
            },
            tf.TensorSpec(shape=[None], dtype=tf.int32),
        ]
    )
    def _sentence_permutation(
        inputs: Dict[str, tf.Tensor], target: tf.Tensor
    ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """Permute by segment token ID"""
        source_tokens = inputs["input_ids"]
        num_source_tokens = tf.cast(tf.shape(source_tokens), tf.int64)

        is_segment = source_tokens == segment_token_id
        segment_end_indices = tf.concat([tf.squeeze(tf.where(is_segment), axis=1), num_source_tokens], axis=0)
        segment_start_indices = tf.concat([[0], segment_end_indices[:-1] + 1], axis=0)
        segment_indices = tf.stack([segment_start_indices, segment_end_indices], axis=1)
        shuffled_segment_indices = tf.random.shuffle(segment_indices)

        first_segment = shuffled_segment_indices[0]
        shuffled_segment_indices = shuffled_segment_indices[1:]
        permutated_source_tokens = source_tokens[first_segment[0] : first_segment[1]]

        num_segments = tf.shape(shuffled_segment_indices)[0]
        for i in tf.range(num_segments):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(permutated_source_tokens, tf.TensorShape([None]))]
            )

            indices = shuffled_segment_indices[i]
            segment = source_tokens[indices[0] : indices[1]]
            permutated_source_tokens = tf.concat([permutated_source_tokens, [segment_token_id], segment], axis=0)

        return {
            "input_ids": permutated_source_tokens,
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": inputs["decoder_input_ids"],
        }, target

    return _sentence_permutation


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
