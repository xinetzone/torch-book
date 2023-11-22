from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch import fx, nn
from torch.nn import functional as F

def replace(model):
    mod = fx.symbolic_trace(model)
    # 遍历 Graph 中全部节点
    for node in mod.graph.nodes:
        # 如果匹配目标
        if node.op == "call_module":
            if "relu" in node.target:
                # 设置插入点，添加新节点，用新节点替换所有 `node` 的用法
                with mod.graph.inserting_after(node):
                    new_node = mod.graph.call_function(F.silu, node.args, node.kwargs)
                    node.replace_all_uses_with(new_node)
                # 移除 graph 中旧的节点
                mod.graph.erase_node(node)
    mod.graph.lint()
    # 不用忘记 recompile！
    new_code = mod.recompile()
    return mod

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(model.conv1.in_channels, 
                        model.conv1.out_channels,
                        3, 1, 1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
model = replace(model)
