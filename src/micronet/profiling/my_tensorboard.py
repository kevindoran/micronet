import logging
import tensorboard
import tensorboard.program
import tensorboard.default
import subprocess


def start_tensorboard(data_dir):
    # Remove http messages.
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    tb = tensorboard.program.TensorBoard(
        tensorboard.default.get_plugins())
        #,tensorboard.default.get_assets_zip_provider())
    tb.configure(argv=[None, '--logdir', data_dir])
    url = tb.launch()
    print('TensorBoard at {}'.format(url))
    return tb


def start_tpu_capture(gcloud_project, tpu_zone, tpu_name, log_dir):
    return subprocess.Popen(['capture_tpu_profile',
                             '--tpu=' + tpu_name,
                             '--tpu_zone=' + tpu_zone,
                             '--duration_ms=' + '3000', # default is 1000
                             '--gcp_project=' + gcloud_project,
                             '--logdir=' + log_dir])


def start_tensorboard_for_tpu(gcloud_project, tpu_zone, tpu_name, data_dir):
    tb = start_tensorboard(data_dir)
    tpu_popen = start_tpu_capture(gcloud_project, tpu_zone, tpu_name, data_dir)
    return (tb, tpu_popen)


def main():
    import sys
    data_dir = sys.argv[1]
    tpu_name = 'kdoran1'
    zone = 'us-central1-f'
    gcloud_project = 'micronet-kdoran'
    tb, popen = start_tensorboard_for_tpu(gcloud_project, zone, tpu_name, data_dir)
    popen.wait()


if __name__ == '__main__':
    main()
