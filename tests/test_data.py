from transformers_bart_training.data import get_dataset


def test_get_dataset():
    dataset = get_dataset("")
    assert next(iter(dataset)) == (0, 0)
