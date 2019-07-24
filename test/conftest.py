import micronet.gcloud
import micronet.estimator
import pytest
import pytz
import datetime


def pytest_addoption(parser):
    parser.addoption('--cloud', action='store_true',
                     help='Signals that the tests are being run in the cloud '
                          'which should be able to run tests on both a CPU and '
                          'TPU.')


# TODO: add a cpu_only mark also.
def pytest_runtest_setup(item):
    if 'tpu_only' in item.keywords and not item.config.getoption('cloud'):
        pytest.skip('This test only runs on cloud TPUs (use --cloud).')


@pytest.fixture
def gcloud_settings(request):
    return micronet.gcloud.load_settings()


@pytest.fixture
def gcloud_temp_path(request, gcloud_settings):
    now = datetime.datetime.now(pytz.timezone('Japan'))
    timestamp_suffix = now.strftime('%Y%m%dT%H%M%S')
    #test_name = request.node.name.replace('.', '_')
    # Previously using node.name, however, more context helps understand the
    # location of the test when viewing the results in tensorflow.
    # test_name = request.node.name
    # Instead, try using the details in node.location. I'm not 100% sure what
    # identifier will be. It is the test name for tests that are not within a
    # class.
    file, line_no, identifier = request.node.location
    temp_path = '{bucket}/pytest/{file}/{test_id}/{timestamp}'.format(
        bucket=gcloud_settings.bucket_url(), file=file, test_id=identifier,
        timestamp=timestamp_suffix)
    temp_path = temp_path.replace('.py', '_py')
    return temp_path


@pytest.fixture
def estimator_fn(request, tmpdir, gcloud_temp_path):
    """Creates and returns a factory that creates a TPUEstimator or Estimator.

    Refer to the nested factory function below for the signature of the returned
    function.

    This fixture is tested in:
        * test_estimator.test_estimator_fixture (for TPUEstimator).
        * FIXME: this also needs a test for the standard Estimator.
    """
    is_cloud = request.config.getoption('--cloud', default=False)
    use_tpu = is_cloud
    if use_tpu:
        model_dir = gcloud_temp_path
        # Don't need this sub-dir (yet).
        # model_dir += '/tensorflow_model'
    else:
        model_dir = str(tmpdir.mkdir('model'))

    # The factory function to be returned:
    def create_estimator(keras_model_fn, train_batch_size, eval_batch_size):
        """Create and return an estimator. Could be a CPU or TPU estimator.

        Args:
            keras_model_fn: a function returning a Keras model.
            train_batch_size: the batch used for training. This will be ignored
                if a CPU estimator is created.
            eval_batch_size: the batch used when evaluating. This will be
                ignored if a CPU estimator is created.

        Returns:
            A tf.Estimator or tf.TPUEstimator.
        """
        if use_tpu:
            model_fn = micronet.estimator.create_model_fn(keras_model_fn,
                    processor_type=micronet.estimator.ProcessorType.TPU)
            gcloud_settings = micronet.gcloud.load_settings()
            estimator = micronet.estimator.create_tpu_estimator(
                gcloud_settings, model_dir, model_fn, train_batch_size,
                eval_batch_size)
        else:
            model_fn = micronet.estimator.create_model_fn(keras_model_fn,
                    processor_type=micronet.estimator.ProcessorType.CPU)
            estimator = micronet.estimator.create_cpu_estimator(model_dir,
                                                                model_fn)
        return estimator

    return create_estimator
