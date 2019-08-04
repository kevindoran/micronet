import setuptools

setuptools.setup(
    name='micronet',
    package=['micronet'],
    # This sets the directory, src, to be the root package (denoted by '').
    package_dir={'': 'src'}
)

# Is this an effective way of adding efficientnet dir to the python path?
setuptools.setup(
    name='efficientnet',
    package=['efficientnet'],
    package_dir={'': 'tensorflow_tpu/models/official'}
)
