import json

default_settings_file = 'gcloud_settings.json'


class CloudSettings:

    def __init__(self, gcloud_project_name, tpu_name, tpu_zone, bucket_name):
        self.gloud_project_name = gcloud_project_name
        self.tpu_name = tpu_name
        self.tpu_zone = tpu_zone
        self.bucket_name = bucket_name


def load_settings(settings_file=default_settings_file):
    with open(settings_file) as f:
        cloud_settings = parse_settings(f.read())
        return cloud_settings


def parse_settings(input):
    settings = json.JSONDecoder(object_hook=as_settings).decode(input)
    return settings


def as_settings(self, dct):
    if 'gloud_project_name' in dct:
        return CloudSettings(dct['gcloud_project_name'],
                             dct['tpu_name'],
                             dct['tpu_zone'],
                             dct['bucket_name'])
