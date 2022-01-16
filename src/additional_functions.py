from os import listdir
from contextlib import suppress


def list_dir(path):
    """
    input: path to directors
    return: sorted list of files and directories

    used to remove unwanted files like '.DS_Store' and any other hidden files
    """

    _list = listdir(path)
    for item in _list:
        if item.startswith("."):
            with suppress(ValueError, AttributeError):
                _list.remove(item)

    _list.sort()
    return _list


def set_paths(parent, _list):
    """
    input:
        parent: path of parent directory
        _list: result of list_dir

    return: given list as merged with parent directory path to have full path
    """

    for i in range(len(_list)):
        _list[i] = parent + "/" + _list[i]

    return _list
