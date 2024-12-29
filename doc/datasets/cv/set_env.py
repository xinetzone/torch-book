import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[3]
print(f"项目根目录：{root_dir}")
sys.path.extend([
    f"{root_dir}/src",
    f"{root_dir}/tests"
])
from env import temp_dir
(temp_dir/"output/datasets").mkdir(exist_ok=True)
