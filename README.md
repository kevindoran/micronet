Install the src folder as a package root in order to run tests and get Pycharm
hints.

    pip install -e . 

Source env:

    source venv/bin/activate

Run all tests:
    
    pytest ./test
    
Run tests and signal that there is a TPU available:

    pytest --cloud ./test    
    
This should be run only on a Google cloud VM with a TPU available in the READY
state. 

Run tests and allow test output to be outputted live to std-out:

    pytest -s ./test



