import micronet.gcloud as gcloud
import google.cloud.storage as gc_storage
import pytest

@pytest.fixture
def test_settings():
    test_settings = gcloud.CloudSettings(
        'micronet',
        'kdoran1',
        'us-central1-f',
        # This is a real bucket, used specifically for testing.:w
        'micronet_test'
    )
    return test_settings


@pytest.fixture
def test_bucket(test_settings):
    storage_client = gc_storage.Client()
    bucket = storage_client.get_bucket(test_settings.bucket_name)
    yield bucket
    for blob in bucket.list_blobs():
        blob.delete()
    assert not len(list(bucket.list_blobs()))


def test_experiment_dir(test_settings, test_bucket, request):
    """
    Tests that the experiment_dir() function determines the correct log dir.

    Useful documentation on how Google Cloud Storage handles directories:
    https://cloud.google.com/storage/docs/gsutil/addlhelp/HowSubdirectoriesWork
    """
    # Setup.
    bucket = test_bucket
    object_data = 'This is a test object created in a pytest test ({}).'.format(
        request.node.name)
    object_name = 'pytest_file.txt'
    base_url = 'gs://{bucket}/'.format(bucket=test_settings.bucket_name)

    dummy_dirs = [
        # models/experiments/
        'models/experiments/1/1',
        'models/experiments/1/2',
        # 'models/experiments/1/3 is created in test 1,
        # 'models/experiments/1/4 is kept empty when testing creation of 5,
        # 'models/experiments/1/5 is created in test 3,
        'models/experiments/1/6',
        'models/experiments/1/6/sub_dir/',
        'models/experiments/1/20',
        'models/experiments/1/21',
        'models/experiments/1/22',
        'models/experiments/1/30',
        'models/experiments/1/50',
        'models/experiments/1/52',
        'models/experiments/1/111',
        'models/experiments/2/1',
        'models/experiments/2/2',
        'models/experiments/2/3',
        'models/experiments/2/4'
    ]
    for d in dummy_dirs:
        blob = test_bucket.blob(blob_name='{}/{}'.format(d, object_name))
        assert not blob.exists()
        blob.upload_from_string(object_data, content_type='text/plain')
        assert blob.exists()

    # Test
    # 1. Basic case. Function returns the expected experiment dir.
    exp_dir = gcloud.experiment_dir(test_settings, 1, 3)
    assert exp_dir == base_url + 'models/experiments/1/3'
    # 2. Previous experiment directory is empty. An exception should be thrown.
    with pytest.raises(Exception):
        gcloud.experiment_dir(test_settings, 1, 5)
    # 3. Previous experiment directory is empty, but skip=True. Same as #1.
    gcloud.experiment_dir(test_settings, 1, 5, allow_skip_minor=True)
    # 4. Experiment directory exists. An exception should be thrown.
    with pytest.raises(Exception):
        gcloud.experiment_dir(test_settings, 1, 2)
    # 5. Experiment directory not empty, but delete_if_exists=True. Same as #1.
    # Insure the dummy file is present, as we expect it to be deleted later.
    exists = bucket.get_blob('models/experiments/1/2/' + object_name) \
             is not None
    assert exists
    gcloud.experiment_dir(test_settings, 1, 2, delete_if_exists=True)
    exists = bucket.get_blob('models/experiments/1/2/' + object_name) \
             is not None
    assert not exists
    # In addition, nothing from experiments/1/20, experiments/1/21, etc should
    # be deleted. If not present, get_blobs() returns None.
    assert bucket.get_blob('models/experiments/1/20/' + object_name) is not None
    assert bucket.get_blob('models/experiments/1/21/' + object_name) is not None
    assert bucket.get_blob('models/experiments/1/22/' + object_name) is not None


