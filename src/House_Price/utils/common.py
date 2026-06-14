import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import yaml


def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return its content as a dictionary.

    This function is used for reading:
    - config/config.yaml
    - config/params.yaml
    - config/schema.yaml
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"YAML file is empty: {path}")

    with open(path, "r", encoding="utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file)

    if content is None:
        raise ValueError(f"YAML file has no valid content: {path}")

    return content


def create_directories(paths: List[str]) -> None:
    """
    Create directories if they do not already exist.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(file_path: str, data: Dict[str, Any]) -> None:
    """
    Save dictionary data as a JSON file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file and return dictionary.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_object(file_path: str, obj: Any) -> None:
    """
    Save Python object using joblib.
    Used for saving model/preprocessor artifacts.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(obj, path)


def load_object(file_path: str) -> Any:
    """
    Load Python object using joblib.
    Used for loading model/preprocessor artifacts.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Object file not found: {path}")

    return joblib.load(path)