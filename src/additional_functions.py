from os import listdir, path
from contextlib import suppress


def list_dir(_path):
    """
        List items in directory and remove unwanted files like '.DS_Store' and any other hidden files.

        @param _path: path of directory.
        @return _list: sorted list of files and directories.
    """

    _list = listdir(_path)
    for item in _list:
        if item.startswith("."):
            with suppress(ValueError, AttributeError):
                _list.remove(item)

    _list.sort()
    return _list


def set_paths(parent, _list):
    """
        @param parent: path of parent directory
        @param _list: result of list_dir

        @return _list: given list as merged with parent directory path to have full path
    """

    for i in range(len(_list)):
        _list[i] = parent + "/" + _list[i]

    return _list


def eliminate_files(_list):
    """
        Removes file paths from given list of file/directory paths

        @param _list: list of file/directory paths
        @return _list: list of directories from the given input
    """
    for item in _list:
        if path.isfile(item):
            _list.remove(item)

    return _list
