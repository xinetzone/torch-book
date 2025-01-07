from pathlib import Path
import sys
# import tools.set_tensorflow
root_dir = Path(__file__).resolve().parents[1]
temp_dir = root_dir/"tests/.temp/tasks"
temp_dir.mkdir(exist_ok=True)
mm_env = [
    f"{root_dir}/doc",
    f"{temp_dir}/mmengine",
    f"{temp_dir}/mmagic",
    f"{temp_dir}/mmdetection",
    f"{temp_dir}/mmpretrain",
    f"{temp_dir}/mmdeploy",
    # f"{temp_dir}/mmcv",
]
sys.path.extend(mm_env)
# print(root_dir)
