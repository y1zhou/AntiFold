# Copyright The Lightning AI team.
#
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
from pathlib import Path

import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "pdb_path": str(Path("data/antibody_antigen/3hfm.pdb").expanduser().resolve()),
        "heavy_chain": "H",
        "light_chain": "L",
        "antigen_chain": "Y",
    },
    timeout=60,
)
print(response.json())
