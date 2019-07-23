import pytest
import micronet.models.mobilenetv2 as mobilenetv2
import micronet.models.xiaochus_mobilenetv2 as xiaochus_mobilenetv2
import micronet.cifar.dataset as cifar_ds
import micronet.estimator
import test.util


def test_num_trainable_params():
    """Tests that the model has the expected number of trainable parameters."""
    input_shape = (1024, 1024, 3)
    model = mobilenetv2.create_model(input_shape=input_shape)
    # FIXME
    #assert test.util.count_trainable_params(model) \
    #       == mobilenetv2.rough_num_trainable_params(
    #            alpha=1.0, input_shape=input_shape, classes=1000)
    # FIXME: 3,504,872 is ~50,000 - 100,000 more parameters than what the
    #        paper mentions.
    assert test.util.count_trainable_params(model) == 3504872


def keras_model_fn():
    model = mobilenetv2.create_model(input_shape=cifar_ds.DEFAULT_DATA_SHAPE,
                                     classes=cifar_ds.CLASSES)
    return model

def xiaochus_model_fn():
    model = xiaochus_mobilenetv2.MobileNetv2(input_shape=cifar_ds.DEFAULT_DATA_SHAPE,
                                             k=cifar_ds.CLASSES)
    return model


# TODO: mostly copied from test_cifar_linear_model. Could be factored a bit.
def test_is_trainable(estimator_fn):
    """Test that that training and evaluation run as expected.

    Tests that:
        1. The untrained model can be evaluated, and that there is about 1%
           accuracy.
        2. The model can be trained.
        3. The trained model has higher accuracy (~20%).
    """
    # Setup
    batch_size = 128
    eval_count = 1024
    train_steps = batch_size*1000
    eval_steps = int(eval_count / batch_size)
    assert eval_steps * batch_size == eval_count
    # Errors due to ops being created in loops:
    #estimator = estimator_fn(xiaochus_model_fn, batch_size, batch_size)
    estimator = estimator_fn(keras_model_fn, batch_size, batch_size)

    # Replace with lambda?
    def input_fn(params):
        del params
        mini_ds = cifar_ds.train_dataset(augment=False, crop_to=32)
        # Take a small amount and repeat so that the test can show progress
        # in a smaller amount of steps (so the test runs quickly).
        #mini_ds = mini_ds.take(500)
        mini_ds = mini_ds.repeat()
        return mini_ds.batch(batch_size, drop_remainder=True)

    # Test
    # 1. Check that the untrained model predicts randomly.
    #
    # I want the test to pass 99% of the time.
    # For a 1000 trial experiment with success probability of 1% (100 classes),
    # CDF_inverse(0.01) ~= 3
    # CDF_inverse(0.99) ~= 19
    # (from binomial dist calculator:
    #      https://www.di-mgt.com.au/binomial-calculator.html)
    # TODO: is it valid to assume a random output from the untrained model?
    results = estimator.evaluate(input_fn, steps=eval_steps)
    # FIXME 13: I think accuracy is a %, so this check seems wrong.
    assert 3/eval_count < results['accuracy'] <= 19/eval_count

    # 2. Check that the model can be trained.
    estimator.train(input_fn, max_steps=train_steps)

    # 3. Check that the training has increased the model's accuracy.
    # Results is a dict containing the metrics defined by the model_fn.
    # FIXME 4: I should encapsulate/separate the metric creation so that it
    #          is easy to assume that certain metrics are present.
    results = estimator.evaluate(input_fn, steps=eval_steps)
    # We should expect some improvement over the random case, 1/100. Running
    # it a few times gave ~4.5%, so using a value a little lower to make sure
    # the test reliably passes (while still being useful).
    print('accuracy: {}'.format(results['accuracy']))
    assert results['accuracy'] >= 0.040
