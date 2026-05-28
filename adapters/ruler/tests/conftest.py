# Copyright [yyyy] eval-hub contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest configuration for RULER adapter tests."""

import os
import sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(os.environ.get("RULER_SCRIPTS_DIR", str(ADAPTER_DIR / "scripts")))

# Adapter root — for `import main` and `from main import ...`
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

# scripts/ — for `import data.template`, `import data.synthetic.constants`
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# scripts/data/ — for `from template import Templates`, `import synthetic.constants`
if str(SCRIPTS_DIR / "data") not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR / "data"))

# scripts/data/synthetic/ — for `from constants import TASKS` (data constants)
if str(SCRIPTS_DIR / "data" / "synthetic") not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR / "data" / "synthetic"))

# scripts/eval/ — for evaluation module imports
if str(SCRIPTS_DIR / "eval") not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR / "eval"))

# scripts/eval/synthetic/ — for `from constants import TASKS` (eval metric functions)
if str(SCRIPTS_DIR / "eval" / "synthetic") not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR / "eval" / "synthetic"))
