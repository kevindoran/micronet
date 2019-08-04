import pytest

# we want to have pytest assert introspection in the helpers
# Solution taken from: https://stackoverflow.com/questions/41522767/pytest-assert-introspection-in-helper-function
pytest.register_assert_rewrite('test.util')