"""This module contains utility functions for loading json and toml files."""

import json
import toml


def load_json(file_path: str) -> dict:
    """Load and parse json file

    Args:
        file_path (str): path to json file

    Returns:
        dict: parsed data
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_toml(file_path: str) -> dict:
    """Load and parse toml file

    Args:
        file_path (str):  path to toml file

    Returns:
        dict: parsed data
    """
    data = toml.load(file_path)
    return data
