import os

import pytest
import tensorflow as tf
import tensorflow_text as text

from transformers_bart_pretrain.data import (
    get_dataset,
    get_tfrecord_dataset,
    make_train_examples,
    sentence_permutation,
    text_infilling,
)

from .const import DEFAULT_SPM_MODEL, TEST_DATA_DIR


def test_get_dataset():
    TEST_SAMPLE_PATH = os.path.join(TEST_DATA_DIR, "sample1.txt")

    with open(DEFAULT_SPM_MODEL, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    dataset = get_dataset(TEST_SAMPLE_PATH, tokenizer, auto_encoding=True, repeat=False)
    examples = [[x.tolist() for x in example] for example in dataset.as_numpy_iterator()]
    assert len(examples) == 3, "Sample dataset example number is wrong"
    assert examples == [
        [[2, 3478, 17, 3], [2, 3478, 17, 3]],
        [[2, 387, 640, 3728, 94, 151, 41, 1412, 4220, 4220, 3], [2, 387, 640, 3728, 94, 151, 41, 1412, 4220, 4220, 3]],
        [[2, 192, 230, 605, 339, 133, 3], [2, 192, 230, 605, 339, 133, 3]],
    ]

    train_examples = make_train_examples(*examples[2])
    tf.debugging.assert_equal(train_examples[0]["input_ids"], [2, 192, 230, 605, 339, 133, 3])
    tf.debugging.assert_equal(train_examples[0]["decoder_input_ids"], [2, 192, 230, 605, 339, 133])
    tf.debugging.assert_equal(train_examples[1], [192, 230, 605, 339, 133, 3])


def test_get_tfrecord_dataset():
    TEST_TFRECORD_PATH = os.path.join(TEST_DATA_DIR, "sample1.tfrecord")

    dataset = get_tfrecord_dataset(TEST_TFRECORD_PATH, False)
    examples = [[x.tolist() for x in example] for example in dataset.as_numpy_iterator()]
    assert len(examples) == 3, "Sample dataset example number is wrong"
    assert examples == [
        [[2, 3478, 17, 3], [2, 3478, 17, 3]],
        [[2, 387, 640, 3728, 94, 151, 41, 1412, 4220, 4220, 3], [2, 387, 640, 3728, 94, 151, 41, 1412, 4220, 4220, 3]],
        [[2, 192, 230, 605, 339, 133, 3], [2, 192, 230, 605, 339, 133, 3]],
    ]

    train_examples = make_train_examples(*examples[2])
    tf.debugging.assert_equal(train_examples[0]["input_ids"], [2, 192, 230, 605, 339, 133, 3])
    tf.debugging.assert_equal(train_examples[0]["decoder_input_ids"], [2, 192, 230, 605, 339, 133])
    tf.debugging.assert_equal(train_examples[1], [192, 230, 605, 339, 133, 3])


@pytest.mark.parametrize("token_length", [12, 10, 25])
def test_text_infilling(token_length):
    text_infilling_fn = text_infilling(-1)

    source_token = tf.random.uniform([token_length], 0, 100, tf.int32)
    dummy = tf.constant([1, 2, 3])
    output = text_infilling_fn(
        {
            "input_ids": source_token,
            "decoder_input_ids": dummy,
            "attention_mask": tf.random.uniform(source_token.shape, 0, 2, dtype=tf.int32),
        },
        dummy,
    )[0]["input_ids"]

    tf.debugging.assert_greater_equal(tf.reduce_sum(tf.cast(output == -1, tf.int32)), 1)
    tf.debugging.assert_less_equal(output.shape[0], token_length + 1)


@pytest.mark.parametrize(
    "sequence_length,max_token_id",
    [(30, 10), (20, 5), (123, 33), (5, 10), (1, 1), (1, 100), (12, 12)],
)
def test_sentence_permutation(sequence_length: int, max_token_id: int):
    input_ids = tf.random.uniform([sequence_length], 0, max_token_id, dtype=tf.int32)
    segment_token_id = tf.random.uniform((), 0, max_token_id, dtype=tf.int32)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": tf.constant([1]),
        "decoder_input_ids": tf.constant([1]),
    }
    target = tf.constant([1])

    sentence_permute_fn = sentence_permutation(segment_token_id)
    for _ in range(30):
        outputs = sentence_permute_fn(inputs, target)
        permuted_input_ids = outputs[0]["input_ids"]

        tf.debugging.assert_equal(outputs[1], target)
        tf.debugging.assert_equal(tf.sort(permuted_input_ids), tf.sort(input_ids))
