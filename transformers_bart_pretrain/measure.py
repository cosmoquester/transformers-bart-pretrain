import tensorflow as tf


class SparseCategoricalCrossentropy(tf.keras.losses.Loss):
    """Normal sparse categorical crossentrophy with ignore index"""

    def __init__(
        self,
        ignore_index: int = 0,
        from_logits=True,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name="sparse_categorical_crossentropy",
    ):
        super().__init__(name=name, reduction=reduction)
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        loss = tf.boolean_mask(loss, y_true != self.ignore_index)
        return loss


class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    """Normal sparse categorical accuracy with ignore index"""

    def __init__(self, ignore_index: int = 0, name="accuracy"):
        super().__init__(name=name)

        self.ignore_index = ignore_index
        self.total_sum = self.add_weight(name="total_sum", initializer="zeros")
        self.total_count = self.add_weight(name="total_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = tf.boolean_mask(accuracy, y_true != self.ignore_index)
        if sample_weight is not None:
            accuracy = tf.multiply(accuracy, sample_weight)

        self.total_sum.assign_add(tf.reduce_sum(accuracy))
        self.total_count.assign_add(tf.cast(tf.shape(accuracy)[0], tf.float32))

        return accuracy

    def result(self):
        return self.total_sum / self.total_count
