import json
import google.cloud.storage as gc_storage
import logging

default_settings_file = 'gcloud_settings.json'
experiments_base_dir = 'models/experiments'


class CloudSettings:

    def __init__(self, project_name, tpu_name, tpu_zone, bucket_name):
        self.project_name = project_name
        self.tpu_name = tpu_name
        self.tpu_zone = tpu_zone
        self.bucket_name = bucket_name

    def bucket_url(self):
        return 'gs://' + self.bucket_name


def load_settings(settings_file=default_settings_file):
    with open(settings_file) as f:
        cloud_settings = parse_settings(f.read())
        return cloud_settings


def parse_settings(input):
    settings = json.JSONDecoder(object_hook=as_settings).decode(input)
    return settings


def as_settings(dct):
    if 'project_name' in dct:
        return CloudSettings(dct['project_name'],
                             dct['tpu_name'],
                             dct['tpu_zone'],
                             dct['bucket_name'])


def experiment_dir(cloud_settings, experiment_major, experiment_minor,
                   delete_if_exists=False, allow_skip_minor=False):
    """Determines the correct path to store an experiment's logs.

    To keep track of different experiments that have been tried, they are
    numbered <major>.<minor>. These ids are referenced in
    documentation/notebooks and in commit messages.

    This function may seem overkill, however, accidentally storing results in
    the wrong directory could misalign results and lead to time-consuming
    misdirected experimentation.

    :param cloud_settings: used to determine which bucket to use.
    :param experiment_major: group ID for set of experiments
    :param experiment_minor: ID of experiment
    :param delete_if_exists: delete all files in the corresponding directory
        if it is not empty.
    :param allow_skip_minor: allow gaps in the minor ID.

    :return (str): the directory to use for logging for the given experiment.
    """
    dir_fmt = experiments_base_dir + '/{major}/{minor}'
    storage_client = gc_storage.Client()
    bucket = storage_client.get_bucket(cloud_settings.bucket_name)
    new_model_dir = dir_fmt.format(major=experiment_major,
                                   minor=experiment_minor)

    def files_exist_in_dir(dir):
        # Add '/' suffix. We use this variable for string matching and don't
        # want a directory like /dir/sub/5 to match /dir/sub/55.
        if dir[-1] != '/':
            dir += '/'
        return len(list(bucket.list_blobs(prefix=dir))) > 0

    if experiment_minor > 1:
        prev_experiment_dir = dir_fmt.format(major=experiment_major,
                                             minor=experiment_minor - 1)
        if not files_exist_in_dir(prev_experiment_dir):
            if not allow_skip_minor:
                raise Exception('Trying to create folder:\n\t{}\n'
                                "yet the previous folder doesn't exist:\n\t{}"
                                .format(new_model_dir, prev_experiment_dir))
            else:
                logging.log(logging.INFO,
                            'No previous log dir ({})\n'
                            'Skipping.'.format(prev_experiment_dir))

    if files_exist_in_dir(new_model_dir):
        if not delete_if_exists:
            raise Exception(
                'Log path already exists:\n\t{}' .format(new_model_dir))
        else:
            blobs_to_delete = bucket.list_blobs(prefix=new_model_dir + '/')
            for b in blobs_to_delete:
                assert b.name.startswith(new_model_dir + '/')
                b.delete()
            assert not files_exist_in_dir(new_model_dir)
    full_url = 'gs://{bucket}/{dir}'.format(bucket=cloud_settings.bucket_name,
                                            dir=new_model_dir)
    return full_url


