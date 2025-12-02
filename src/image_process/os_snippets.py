import os

def remove_empty_folders(path_abs):
    """Remove empty subdirectories given a parent directory."""
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)

def try_mkdir(d):
    """Try to make a new directory d. Passes if it already exists."""
    try:
        os.makedirs(d)
    except FileExistsError:
        print(f'{d} exists.')
        pass