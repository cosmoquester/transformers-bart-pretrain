import os

import tensorflow as tf
import tensorflow_text as text

from transformers_bart_training.data import get_dataset, get_tfrecord_dataset, make_train_examples

from .const import DEFAULT_SPM_MODEL, TEST_DATA_DIR


def test_get_dataset():
    TEST_SAMPLE_PATH = os.path.join(TEST_DATA_DIR, "sample1.txt")

    with open(DEFAULT_SPM_MODEL, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read(), add_bos=True, add_eos=True)

    dataset = get_dataset(TEST_SAMPLE_PATH, tokenizer, auto_encoding=True)
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

    dataset = get_tfrecord_dataset(TEST_TFRECORD_PATH)
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
