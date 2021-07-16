import os

import pytest
import tensorflow as tf
import tensorflow_text as text

from transformers_bart_pretrain.data import get_dataset, get_tfrecord_dataset, make_train_examples, text_infilling

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
    output = text_infilling_fn({"input_ids": source_token, "decoder_input_ids": dummy}, dummy)[0]["input_ids"]

    tf.debugging.assert_greater_equal(tf.reduce_sum(tf.cast(output == -1, tf.int32)), 1)
    tf.debugging.assert_less_equal(output.shape[0], token_length + 1)
