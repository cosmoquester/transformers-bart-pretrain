import tensorflow as tf


def get_dataset(dataset_file_path: str):
    """
    Read dataset file and construct tensorflow dataset

    :param dataset_file_path: dataset file path.
    """
    # Replace the context of this function
    dataset_x = tf.data.Dataset.range(10000).map(lambda x: x[tf.newaxis])
    dataset_y = tf.data.Dataset.from_tensor_slices([0, 1] * 5000)

    return tf.data.Dataset.zip((dataset_x, dataset_y))
