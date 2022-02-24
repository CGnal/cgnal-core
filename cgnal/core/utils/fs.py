"""Module for basic functionalities to operate with local file systems."""

import errno
import os

from cgnal.core.typing import PathLike


def mkdir(path: PathLike) -> None:
    """
    Create a dir, using a formulation consistent between 2.x and 3.x python versions.

    :param path: path to create
    :return: None
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def create_dir_if_not_exists(directory: PathLike) -> PathLike:
    """
    Create a directory if it does not exist.

    :param directory: path
    :return: directory, str
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_lexicographic_dirname(dirpath: PathLike, first: bool = False) -> PathLike:
    """
    Return the first (or last) subdirectory name ordered lexicographically.

    :param dirpath: name of the base path whose subdirectories ought to be listed
    :param first: whether the first or the last element should be returned
    :return: first (or last) subdirectory name ordered lexicographically.
    """
    return sorted(
        [
            os.path.join(dirpath, o).split("/")[-1]
            for o in os.listdir(dirpath)
            if os.path.isdir(os.path.join(dirpath, o))
        ],
        key=str.lower,
    )[0 if first else -1]
