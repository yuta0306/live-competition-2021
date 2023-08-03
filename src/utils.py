import re
from pathlib import Path
from typing import Any, Dict, Union


def read_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    if not isinstance(file_path, (str, Path)):
        raise ValueError

    with open(file=file_path, mode="r", encoding="utf-8") as f:
        content = f.readlines()

    config = {}
    for line in content:
        if "=" in line:
            line = re.sub(r"\s", "", line)
            key, value_ = line.split("=")
            key = str(_convert_type(key))
            value = _convert_type(value_)
            config.update({key: value})

    return config


def _convert_type(value: str) -> str | int | float:
    if re.fullmatch(r"\d+", value):
        return int(value)
    elif re.fullmatch(r"\d+\.\d+", value):
        return float(value)

    return value
