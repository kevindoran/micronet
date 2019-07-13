import pytest
import tensorflow as ft
import micronet.cifar.linear_model as cifar_linear_model
import micronet.cifar.dataset as cifar_ds
import micronet.estimator
import functools


#@pytest.mark.skip(reason="Not finished yet.")
def test_is_trainable(tmpdir):
    # get a small data subset and repeat
    # train
    # test
    # check that accuracy is 100 %
    # tmpdir is from the 'py' package. tmpdir.mkdir() creates a py.localPath
    # which is convertible to a string via str().
    max_steps = 1000
    model_dir = str(tmpdir.mkdir("model"))
    num_records = 100

    # Replace with lambda?
    def train_input_fn(params):
        del params
        mini_ds = cifar_ds.train_dataset(augment=False).take(num_records)
        return mini_ds.repeat().batch(5, drop_remainder=True)

    def eval_input_fn(params):
        del params
        # The processing done by cifar_ds.train_dataset() needs to be done in
        # the same graph as the training evaluation etc. Thus, it must be
        # placed within the input functions, and not done outside. Hence the
        # line below is duplicated in both input functions.
        mini_ds = cifar_ds.train_dataset(augment=False).take(num_records)
        return mini_ds.batch(5, drop_remainder=True)

    processor = micronet.estimator.ProcessorType.CPU
    model_fn = functools.partial(cifar_linear_model.model_fn, processor)
    estimator = micronet.estimator.create_cpu_estimator(model_dir, model_fn)
    estimator.train(train_input_fn, max_steps=100)
    # Results is a dict containing the metrics defined by the model_fn.
    # FIXME 4: I should encalpusate/separate the metric creation so that it
    # is easy to assume that certain metrics are present.
    results = estimator.evaluate(eval_input_fn, steps=100)
    # We should expect some improvement over the random case, 1/100.
    assert results['accuracy'] >= 0.20
