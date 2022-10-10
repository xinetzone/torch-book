import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# TVM_ROOT = "../src"
# # print(TVM_ROOT)
# set_tvm(TVM_ROOT)
sys.path.extend([f"{ROOT}/src"])
