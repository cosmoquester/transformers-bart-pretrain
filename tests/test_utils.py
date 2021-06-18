import numpy as np
import pytest

from sample_package.utils import LRScheduler


@pytest.mark.parametrize(
    "num_epoch,learning_rate,min_learning_rate,warmup_rate",
    [(10, 1.1, 0.0, 0.3), (33, 1e-5, 1e-7, 0.1), (100, 100, 0, 0.5)],
)
def test_learning_rate_scheduler(num_epoch, learning_rate, min_learning_rate, warmup_rate):
    fn = LRScheduler(num_epoch, learning_rate, min_learning_rate, warmup_rate)

    for i in range(num_epoch):
        learning_rate = fn(i)
    np.isclose(learning_rate, min_learning_rate, 1e-10, 0)
