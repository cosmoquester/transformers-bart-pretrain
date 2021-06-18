import pytest
import tensorflow as tf

from transformers_bart_training.model import SampleModel


def test_model():
    model = SampleModel(15)
    model(tf.constant([[10, 11, 12]]))

    with pytest.raises(Exception):
        model(tf.constant([[30]]))
