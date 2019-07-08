# TODO: move into separate file.
# Copied from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tpu/profiler/capture_tpu_profile.py
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as resolver
from tensorflow.python.eager import profiler_client
from tensorflow.python.framework import errors


def tensorboard_init_for_tpus():
    tpu_cluster_resolver = resolver.TPUClusterResolver(
        tpu_name, zone=tpu_zone, project=gcloud_project_name)
    service_addr = tpu_cluster_resolver.get_master()
    service_addr = service_addr.replace('grpc://', '').replace(':8470', ':8466')
    duration_ms = 1000
    workers_list = get_workers_list(tpu_cluster_resolver)
    include_dataset_ops = True # Use False for longer TPU traces.
    num_tracing_attempts = 3
    profiler_client.start_tracing(service_addr, model_dir, duration_ms,
                                  workers_list, include_dataset_ops,
                                  num_tracing_attempts)


def get_workers_list(cluster_resolver):
    """Returns a comma separated list of TPU worker IP addresses.

    Gets cluster_spec from cluster_resolver. Use the worker's task indices to
    obtain and return a list of ip addresses.

    Args:
      cluster_resolver: TensorFlow TPUClusterResolver instance.

    Returns:
      A string of comma separated list of IP addresses. For example:
      '10.2.0.1,10.2.0.2,10.2.0.3,10.2.0.4'

    Raises:
      UnavailableError: cluster_resolver doesn't contain a valid cluster_spec.
    """
    worker_job_name = 'worker'
    cluster_spec = cluster_resolver.cluster_spec()
    if not cluster_spec:
      raise errors.UnavailableError(
          'None', 'None',
          'Cluster spec not found, your client must run in GCE environment.')
    task_indices = cluster_spec.task_indices(worker_job_name)
    workers_list = [
        cluster_spec.task_address(worker_job_name, i).split(':')[0]
        for i in task_indices
    ]
    return ','.join(workers_lis