import json
import google.cloud.storage as gc_storage
import logging
from contextlib import contextmanager
import subprocess
import re
import enum

default_settings_file = 'gcloud_settings.json'
experiments_base_dir = 'models/experiments'
tpu_name_prefix = 'auto_tpu_'
max_tpus = 20

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


DirExistsBehaviour = enum.Enum('DirExistsBehaviour', 'FAIL OVERWRITE CONTINUE')
def experiment_dir(cloud_settings, experiment_major, experiment_minor,
                   dir_exists_behaviour=DirExistsBehaviour.FAIL,
                   allow_skip_minor=False):
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
        if dir_exists_behaviour == DirExistsBehaviour.FAIL:
            raise Exception(
                'Log path already exists:\n\t{}' .format(new_model_dir))
        elif dir_exists_behaviour == DirExistsBehaviour.OVERWRITE:
            blobs_to_delete = bucket.list_blobs(prefix=new_model_dir + '/')
            for b in blobs_to_delete:
                assert b.name.startswith(new_model_dir + '/')
                b.delete()
            assert not files_exist_in_dir(new_model_dir)
        elif dir_exists_behaviour == DirExistsBehaviour.CONTINUE:
            pass
        else:
            raise Exception('Unexpected DirExistsBehaviour: {}'
                            .format(dir_exists_behaviour))
    full_url = 'gs://{bucket}/{dir}'.format(bucket=cloud_settings.bucket_name,
                                            dir=new_model_dir)
    return full_url


@contextmanager
def start_tpu(project, zone, description=None):
    # We could use the Google cloud compute Python library, but it it's the
    # most use friendly and it's not clear what TPU support there is.
    completed_cmd = subprocess.run(
        ['gcloud', 'compute', 'tpus', 'list', '--format', 'json'],
        # Only available in Python 3.7
        # capture_output=True)
        # 3.5 version:
        stdout=subprocess.PIPE,
        check=True,
        # The next line is needed to capture output as str (not byte) in 3.5.
        universal_newlines=True)
    tpu_info = json.loads(completed_cmd.stdout)
    id_, already_created = _get_free_tpu_id(tpu_info, project, zone)
    tpu_name = tpu_name_prefix + str(id_)
    if already_created:
        cmd = _tpu_start_cmd(tpu_name, project, zone)
    else:
        network = _tpu_network(id_)
        cmd = _tpu_create_cmd(tpu_name, network, project, zone, description)
    res = subprocess.run(cmd, check=True)
    assert res.returncode == 0
    try:
        yield tpu_name
    finally:
        cmd = _tpu_stop_cmd(tpu_name, zone)
        subprocess.run(cmd, check=True)


def _get_free_tpu_id(tpu_list_output, project_name, zone):
    tpu_name_pattern = \
        'projects/{project_name}/locations/{location}/nodes/{prefix}(\d+)' \
            .format(project_name=project_name,
                    location=zone,
                    prefix=tpu_name_prefix)
    used_ids = set()
    live_tpus = 0
    free_id = None
    for tpu in tpu_list_output:
        if tpu['state'] != 'STOPPED':
            live_tpus += 1
        name = tpu['name']
        m = re.match(tpu_name_pattern, name)
        if not m:
            # This TPU was not created by gcloud. Ignore.
            continue
        id = int(m.group(1))
        if tpu['state'] == 'STOPPED':
            # We can use this TPU. It doesn't need to be created.
            if 'health' not in tpu:
                # Sometimes 'health' isn't present. It's not clear why the
                # health key is sometimes not present. So far, it's only been
                # seen for TPUs that were subsequently started successfully.
                logging.warning('Found an available TPU, but its health status '
                                'is not listed.')
            elif tpu['health'] != 'HEALTHY':
                logging.warning('Found an available TPU, but it is not healthy'
                                '. ({})'.format(tpu['state']))
            free_id = (id, True)
            break
        else:
            used_ids.add(id)
    if not free_id:
        assert live_tpus <= max_tpus
        num_available = max_tpus - live_tpus
        assert num_available >= 0
        if not num_available:
            raise Exception('Max TPU count reached ({}).'.format(max_tpus))
        i = 0
        while not free_id:
            if i not in used_ids:
                # Found a free ID. This TPU will need to be created first.
                free_id = (i, False)
            i += 1
    assert free_id
    id = free_id[0]
    already_created = free_id[1]
    return id, already_created


def _tpu_network(tpu_id: int):
    """Map a TPU ID to a network address.

    This is just a handy way to insure that there is a subnet available.
    """
    network = '10.128.17.{}/29'.format(8 * tpu_id)
    return network


def _tpu_start_cmd(name, project, zone):
    cmd = 'gcloud compute tpus start {tpu_name} ' \
              '--project={project} ' \
              '--zone={zone} '.format(
                    tpu_name=name,
                    project=project,
                    zone=zone)
    return cmd.split()


def _tpu_create_cmd(name, network, project, zone, description):
    full_desc = 'TPU created by {}.'.format(__file__) if __file__ \
        else 'TPU created in interactive session.'
    if description:
        full_desc += '\n' + description
    cmd = 'gcloud compute tpus create {tpu_name} ' \
              '--project={project} '               \
              '--zone={zone} '                     \
              '--network=default '                 \
              '--range={network} '                 \
              '--version=1.14 '                    \
              '--preemptible '                     \
              '--accelerator-type=v2-8 '.format(
                    tpu_name=name,
                    project=project,
                    zone=zone,
                    network=network)
    cmd_as_list = cmd.split()
    # Add the description last, at is may have spaces.
    cmd_as_list.append('--description="{}"'.format(description))
    return cmd_as_list


def _tpu_stop_cmd(tpu_name, zone):
    cmd = 'gcloud compute tpus stop {tpu_name} --zone {zone}'.format(
        tpu_name=tpu_name, zone=zone)
    return cmd.split()

