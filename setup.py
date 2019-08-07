import setuptools

setuptools.setup(
    name='micronet',
    packages=['micronet'],
    # This sets the directory, src, to be the root package (denoted by '').
    # It should be possible to do this:
    # package_dir={'micronet': 'src/micronet'}
    # But this doesn't work with editable installs, as described here.
    # https://stackoverflow.com/questions/19602582/pip-install-editable-links-to-wrong-path
    # Possibly related to the  long open bug:
    # https://github.com/pypa/setuptools/issues/230
    # So, the package must be added at the root level.
    package_dir={'': 'src'}
)

# Is this an effective way of adding efficientnet dir to the python path?
# To allow 'import efficientnet':
setuptools.setup(
    name='efficientnet',
    package=['efficiennet'],
    package_dir={'': 'tensorflow_tpu/models/official'}
)

# As efficientnet isn't actually a package, and the modules within it assume
# that their directory is on sys.path, they import other modules within their
# directory like 'import preprocessing'. Thus, if we need to make these modules
# available without simply placing them in an efficientnet package.
# Note that this will install a very generic modules 'utils' into the root
# package.
setuptools.setup(
    name='efficientnet_internal',
    # Setting the package to '' makes the modules in the package_dir be
    # installed in the root package (i.e. the modules are accessible via imports
    # without prefixes.
    package=[''],
    # The modules will be going into the root packages, which is not ideal, as
    # they could cause unexpected namespace issues.
    # It would be nice to be able to use py_modules to list the individual
    # modules to use, however, I haven't been able to get this working with
    # editable installs.
    package_dir={'': 'tensorflow_tpu/models/official/efficientnet'}
)
