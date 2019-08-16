import micronet.gcloud as gcloud
import google.cloud.storage as gc_storage
import pytest
import json
import copy

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


def test_get_free_tpu_id():
    """Tests that _get_free_tpu_id() finds an unused TPU id."""
    tpu_list_output_example = """
    [{                 
    "health": "HEALTHY", 
    "name": "projects/dummy_project/locations/us-central1-f/nodes/auto_tpu_0",  
    "state": "READY"  
    }, {                 
    "health": "HEALTHY",
    "name": "projects/dummy_project/locations/us-central1-f/nodes/kdoran1",
    "state": "STOPPED"
    }, {
    "health": "HEALTHY", 
    "name": "projects/dummy_project/locations/us-central1-f/nodes/auto_tpu_1",  
    "state": "READY"
    }, {
    "health": "HEALTHY",
    "name": "projects/dummy_project/locations/us-central1-f/nodes/auto_tpu_2",
    "state": "READY"
    }, {
    "health": "HEALTHY",
    "name": "projects/dummy_project/locations/us-central1-f/nodes/auto_tpu_4",
    "state": "READY"
    }]"""
    project = 'dummy_project'
    zone = 'us-central1-f'
    output_as_list = json.loads(tpu_list_output_example)

    # Test
    # 1. No STOPPED TPUs available.
    free_id, created = gcloud._get_free_tpu_id(output_as_list,
                                               project, zone)
    assert free_id == 3
    assert not created

    # 2. Stopped TPU available.
    output_copy = copy.deepcopy(output_as_list)
    output_copy[3]['state'] = 'STOPPED'
    free_id, created = gcloud._get_free_tpu_id(output_copy, project, zone)
    assert free_id == 2
    assert created

    # 3. Don't exceed the TPU limit.
    name = 'projects/dummy_project/locations/us-central1-f/nodes/auto_tpu_{}'
    tpu_list = []
    for i in range(gcloud.max_tpus - 1):
        tpu_list.append(
            {
                'name': name.format(i),
                'state': 'READY'
            }
        )
    free_id, created = gcloud._get_free_tpu_id(tpu_list, project, zone)
    assert free_id == gcloud.max_tpus - 1
    # Once the limit is reached, exception should be thrown.
    tpu_list.append(
        {
            'name': name.format(gcloud.max_tpus),
            'state': 'READY'
        }
    )
    with pytest.raises(Exception):
        gcloud._get_free_tpu_id(free_id, project, zone)
