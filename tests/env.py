from pathlib import Path
import sys
# import tools.set_tensorflow
root_dir = Path(__file__).resolve().parent
temp_dir = root_dir/".temp/tasks"
temp_dir.mkdir(exist_ok=True)
sys.path.extend([
    f"{temp_dir}/mmengine",
    f"{temp_dir}/mmcv",
    f"{temp_dir}/mmagic",
    f"{temp_dir}/mmdetection",
    f"{temp_dir}/mmpretrain",
    f"{temp_dir}/mmdeploy",
])
# print(root_dir)
