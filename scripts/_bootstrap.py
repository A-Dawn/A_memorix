from __future__ import annotations

import sys
import types
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = CURRENT_DIR.parent
if PLUGIN_ROOT.parent.name == "src":
    SRC_ROOT = PLUGIN_ROOT.parent
    PROJECT_ROOT = SRC_ROOT.parent
else:
    SRC_ROOT = PLUGIN_ROOT
    PROJECT_ROOT = PLUGIN_ROOT
WORKSPACE_ROOT = PROJECT_ROOT
MAIBOT_ROOT = PROJECT_ROOT

for _path in (SRC_ROOT, PROJECT_ROOT, PLUGIN_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

if "A_memorix" not in sys.modules:
    package = types.ModuleType("A_memorix")
    package.__path__ = [str(PLUGIN_ROOT)]
    sys.modules["A_memorix"] = package

from A_memorix.paths import config_path, default_data_dir, resolve_repo_path

DEFAULT_CONFIG_PATH = config_path()
DEFAULT_DATA_DIR = default_data_dir()
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "MaiBot.db"
