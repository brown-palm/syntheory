import os
import contextlib

@contextlib.contextmanager
def use_770_permissions():
    """Using this context manager ensures we write files with 770 permissions. 
    """
    original_umask = os.umask(0o007)
    try:
        yield
    finally:
        os.umask(original_umask)


@contextlib.contextmanager
def no_output():
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        yield