from collections import namedtuple
import argparse
import micronet.gcloud as gcloud


Arguments = namedtuple('Arguments', ['allow_skip_patch', 'target_tpu',
                                     'overwrite', 'continue_training',
                                     'dir_exists_behaviour'])


def parse_args(cmd_description: str = '') -> Arguments:
    """Creates and runs an ArgumentParser.

    cmd_description: the script description to be shown by the argument parser.
    """
    parser = argparse.ArgumentParser(description=cmd_description)
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Delete and replace any existing logs for this '
                             'experiment.')
    parser.add_argument('-c', '--continue-training', action='store_true',
                        help='Continue from a previous checkpoint if logs are'
                             ' already present.'
                             'experiment.')
    parser.add_argument('-s', '--allow-skip', action='store_true',
                        help='Allow skipping experiment patch versions.')
    parser.add_argument('-t', '--tpu', type=str, required=False,
                        help='Use a specific TPU.')
    args = parser.parse_args()
    overwrite = args.overwrite
    continue_training = args.continue_training
    if overwrite:
        dir_exists_behaviour = gcloud.DirExistsBehaviour.OVERWRITE
    elif continue_training:
        dir_exists_behaviour = gcloud.DirExistsBehaviour.CONTINUE
    else:
        dir_exists_behaviour = gcloud.DirExistsBehaviour.FAIL
    if overwrite and continue_training:
        raise Exception('Either --overwrite or --continue-training may be '
                        'provided, but not both')
    allow_skip_patch = args.allow_skip
    target_tpu = args.tpu
    return Arguments(allow_skip_patch, target_tpu, overwrite, continue_training,
                     dir_exists_behaviour)
