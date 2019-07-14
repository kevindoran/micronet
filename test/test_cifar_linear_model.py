import pytest
import micronet.cifar.linear_model as cifar_linear_model
import micronet.cifar.dataset as cifar_ds
import micronet.estimator
import test.common


def test_num_trainable_params():
    """Tests that the model has the expected number of trainable parameters."""
    model = micronet.cifar.linear_model.create_model()
    assert test.common.count_trainable_params(model) \
           == cifar_linear_model.NUM_TRAINABLE_PARAM
    # Just for sanity sake, so that I know the true value and if it changes:
    assert cifar_linear_model.NUM_TRAINABLE_PARAM == 172900


def test_is_trainable(tmpdir):
    """Test that that training and evaluation run as expected.

    Tests that:
        1. The untrained model can be evaluated, and that there is about 1%
           accuracy.
        2. The model can be trained.
        3. The trained model has higher accuracy (~20%).
    """
    # Setup
    model_dir = str(tmpdir.mkdir("model"))
    batch_size = 5
    eval_count = 1000
    eval_steps = eval_count / batch_size
    assert eval_steps == int(eval_steps)
    model_fn = micronet.cifar.linear_model.create_model_fn(
        processor_type=micronet.estimator.ProcessorType.CPU)
    estimator = micronet.estimator.create_cpu_estimator(model_dir, model_fn)

    # Replace with lambda?
    def input_fn(params):
        del params
        mini_ds = cifar_ds.train_dataset(augment=False)
        # Take a small amount and repeat so that the test can show training
        # in a smaller amount of steps (so the test runs quickly).
        mini_ds.take(500).repeat()
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
    assert 3/eval_count < results['accuracy'] <= 19/eval_count

    # 2. Check that the model can be trained.
    # Using the eval_steps as the max training steps. Could use something else.
    estimator.train(input_fn, max_steps=eval_steps)

    # 3. Check that the training has increased the model's accuracy.
    # Results is a dict containing the metrics defined by the model_fn.
    # FIXME 4: I should encapsulate/separate the metric creation so that it
    #          is easy to assume that certain metrics are present.
    results = estimator.evaluate(input_fn, steps=eval_steps)
    # We should expect some improvement over the random case, 1/100. Running
    # it a few times gave ~4.5%, so using a value a little lower to make sure
    # the test reliably passes (while still being useful).
    assert results['accuracy'] >= 0.040
