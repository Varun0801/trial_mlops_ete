import sys

def is_venv():
    return sys.prefix != sys.base_prefix

if is_venv():
    print("You are inside a virtual environment.")
else:
    print("You are not inside a virtual environment.")