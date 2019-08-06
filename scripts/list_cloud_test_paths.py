#!/usr/bin/env python

"""Program to list log directories on Google Cloud storage in such a way that
the paths are compatible with tensorboard's 'logdir' option."""

import datetime
import pytz
import google.cloud.storage as gc_storage
import argparse
import re


def now():
    return datetime.datetime.now(pytz.timezone('Japan'))


def find_model_dirs(from_datetime, to_datetime=None, str_filter=None):
    if not to_datetime:
        to_datetime = now()
    bucket_name = 'micronet_bucket1'
    timestamp_pattern = r'^.*(20\d{6}T\d{6})/$'  # example: 20191225T130559/
    timestamp_regex = re.compile(timestamp_pattern)
    storage_client = gc_storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    # Find all matching folders.
    # Oh, how the Google Storage API is lacking.
    test_dir = 'pytest'
    # Blobs are all filenames at all sub-levels.
    blobs = bucket.list_blobs(prefix=test_dir)
    matching_folders = set()
    for b in blobs:
        if str_filter and not re.search(str_filter, b.name):
            continue
        # Have to manually find the folders, due to lacking API support.
        # We know the folder objects have a format like:
        # pytest/<python_file_name>/<test_name>/20191225T130559/
        # Just search for paths ending with a timestamp.
        m = timestamp_regex.match(b.name)
        if not m:
            continue
        # If we have gotten here, b is a folder for an individual pytest run.
        # Filter out any that aren't in the desired date range.
        # We could use the timestamp in the path, but it seems easier to just
        # use the object creation datetime.
        creation_datetime = b.time_created
        if not (from_datetime <= creation_datetime <= to_datetime):
            continue
        # Match! Create a path usable by tensorboard.
        # Use the path timestamp as the tensorboard 'name' assigned to the log
        # dir. This makes it easy to match tensorboard results to folder paths.
        prefix = m.group(1)
        full_path = '{}:gs://{}/{}'.format(prefix, bucket_name, b.name)
        matching_folders.add(full_path)
    return matching_folders


def print_model_dirs(dirs, delimitator=','):
    print(delimitator.join(dirs))


def go_back(days=0, hours=0, minutes=0):
    return now() - datetime.timedelta(days=days, hours=hours, minutes=minutes)


def main():
    # Change the formatting of the help text to allow newlines. Required by
    # the epilog option.
    class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter): pass
    parser = argparse.ArgumentParser(
        description='List Tensorflow model directories outputted during pytest '
                    'runs *today*.',
        epilog='Example:\n'
               '1. Start tensorboard will all logs from the last day '
               '(24 hours).\n'
               "tensorboard --logdir=$({} -d 1)".format(__file__) +
                '\n\n'
                '2. Start tensorboard will all logs from the last hour.\n'
                "tensorboard --logdir=$({} --hours 1)".format(__file__),
        formatter_class=Formatter
    )
    parser.add_argument('-f', '--filter',
                        help='Include only paths matching regex <FILTER>.',
                        type=str,
                        required=False)
    parser.add_argument('-d', '--days',
                        help='go back <DAYS> days in logs (in addition to '
                             'minutes and hours)',
                        type=float,
                        required=False,
                        default=0)
    parser.add_argument('-o', '--hours',
                        help='go back <HOURS> hours in logs (in addition to '
                             'days and minutes)',
                        type=float,
                        required=False,
                        default=0)
    parser.add_argument('-m', '--minutes',
                        help='go back <MINUTES> minutes in logs (in addition to'
                             ' days and minutes)',
                        type=float,
                        required=False,
                        default=0)
    args = parser.parse_args()

    # Run
    print_model_dirs(
        find_model_dirs(go_back(args.days, args.hours, args.minutes),
                        str_filter=args.filter))


if __name__ == '__main__':
    main()
