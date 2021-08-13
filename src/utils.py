from pathlib import Path
import re

from typing import Any, Dict, Union


def read_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    if not isinstance(file_path, (str, Path)):
        raise ValueError

    with open(file=file_path, mode='r', encoding='utf-8') as f:
        content = f.readlines()

    config = {}
    for line in content:
        if '=' in line:
            line = re.sub('\s', '', line)
            key, value = line.split('=')
            key = _convert_type(key)
            value = _convert_type(value)
            config.update({key: value})
    
    return config

def _convert_type(value: str):
    if re.fullmatch('\d+', value):
        value = int(value)
    elif re.fullmatch('\d+\.\d+', value):
        value = float(value)

    return value