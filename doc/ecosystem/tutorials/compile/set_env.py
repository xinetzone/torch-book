# from pathlib import Path
import os
from tvm_book.config.env import set_tvm
# TVM_ROOT = Path(__file__).resolve().parents[5]
# print(TVM_ROOT)
TVM_ROOT = "/media/pc/data/lxw/ai/tvm/"
# print(TVM_ROOT)
set_tvm(TVM_ROOT)
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
