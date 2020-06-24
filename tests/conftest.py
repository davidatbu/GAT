"""PyTest fixtures that are shared across tests files.

https://docs.pytest.org/en/stable/fixture.html#conftest-py-sharing-fixture-functions

    "If during implementing your tests you realize that you want to use a fixture
     function from multiple test files you can move it to a conftest.py file. You don’t
     need to import the fixture you want to use in a test, it automatically gets
     discovered by pytest. The discovery of fixture functions starts at test classes,
     then test modules, then conftest.py files and finally builtin and third party
     plugins."

"""
from __future__ import annotations

import shutil
import tempfile
import typing as T
from pathlib import Path

import torch

from Gat import config
from Gat import data
from Gat.neural import layers
