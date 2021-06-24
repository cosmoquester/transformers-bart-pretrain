import os

import tensorflow as tf
import tensorflow_text as text

from transformers_bart_training.data import get_dataset, make_train_examples

from .const import DEFAULT_SPM_MODEL, TEST_DATA_DIR


def test_get_dataset():
    TEST_SAMPLE_PATH = os.path.join(TEST_DATA_DIR, "sample1.txt")

    with open(DEFAULT_SPM_MODEL, "rb") as f:
        tokenizer = text.SentencepieceTokenizer(f.read())

    dataset = get_dataset(TEST_SAMPLE_PATH, tokenizer, auto_encoding=True)
    examples = [[x.tolist() for x in example] for example in dataset.as_numpy_iterator()]
    assert len(examples) == 3, "Sample dataset example number is wrong"
    assert examples == [
        [[3478, 17], [3478, 17]],
        [[387, 640, 3728, 94, 151, 41, 1412, 4220, 4220], [387, 640, 3728, 94, 151, 41, 1412, 4220, 4220]],
        [[192, 230, 605, 339, 133], [192, 230, 605, 339, 133]],
    ]

    train_examples = make_train_examples(*examples[2])
    tf.debugging.assert_equal(train_examples[0]["input_ids"], [192, 230, 605, 339, 133])
    tf.debugging.assert_equal(train_examples[0]["decoder_input_ids"], [192, 230, 605, 339])
    tf.debugging.assert_equal(train_examples[1], [230, 605, 339, 133])
