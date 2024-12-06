import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]
sys.path.extend([
    f"{root_dir}/src",
    f"{root_dir}/tests"
])
from set_env import temp_dir
(temp_dir/"output").mkdir(exist_ok=True)
