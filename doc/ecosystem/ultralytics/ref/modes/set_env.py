import sys
_path = "/media/pc/data/lxw/ai/ultralytics" # ultralytics api 所在目录
sys.path.append(_path)
from pathlib import Path
from ultralytics import settings
temp_dir = Path("../.temp") # 设置缓存目录
temp_dir.mkdir(exist_ok=True, parents=True)
# 更新项目配置
settings.update({'weights_dir': f'{temp_dir}/weights'})
# pip install onnxruntime-gpu tensorrt -i https://pypi.tuna.tsinghua.edu.cn/simple