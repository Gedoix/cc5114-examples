import os
import shutil
import stat


# http://stackoverflow.com/questions/1889597/deleting-directory-in-python
def _remove_readonly(fn, path_, _):
    # Handle read-only files and directories
    if fn is os.rmdir:
        os.chmod(path_, stat.S_IWRITE)
        os.rmdir(path_)
    elif fn is os.remove:
        os.lchmod(path_, stat.S_IWRITE)
        os.remove(path_)
    return


def force_remove_file_or_symlink(path_):
    try:
        os.remove(path_)
    except OSError:
        os.lchmod(path_, stat.S_IWRITE)
        os.remove(path_)
    return


# Code from shutil.rmtree()
def is_regular_dir(path_):
    try:
        mode = os.lstat(path_).st_mode
    except os.error:
        mode = 0
    return stat.S_ISDIR(mode)


def clear_dir(path_):
    if is_regular_dir(path_):
        # Given path is a directory, clear its content
        for name in os.listdir(path_):
            full_path = os.path.join(path_, name)
            if is_regular_dir(full_path):
                shutil.rmtree(full_path, onerror=_remove_readonly)
            else:
                force_remove_file_or_symlink(full_path)
    else:
        # Given path is a file or a symlink.
        # Raise an exception here to avoid accidentally clearing the content
        # of a symbolic linked directory.
        raise OSError("Cannot call clear_dir() on a symbolic link")
    return
