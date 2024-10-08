# Yacs 配置

Detectron2 提供了基于键值的配置系统，可用于获取标准的常见行为。

本系统使用 YAML 和 [yacs](https://github.com/rbgirshick/yacs)。YAML 是一种非常有限的语言，所以不期望 detectron2 中的所有功能都可以通过配置来实现。如果您需要一些配置空间中无法提供的东西，请使用 detectron2 的 API 编写代码。

随着更强大的 [LazyConfig](lazyconfigs) 系统的引入，Detectron2 不再向基于 Yacs/YAML 的配置系统添加功能/新键。

## Yacs 基本用法

这里展示了 {class}`detectron2.config.CfgNode` 对象的一些基本用法。

```python
from detectron2.config import get_cfg
cfg = get_cfg()    # obtain detectron2's default config
cfg.xxx = yyy      # add new configs for your own custom components
cfg.merge_from_file("my_cfg.yaml")   # load values from a file

cfg.merge_from_list(["MODEL.WEIGHTS", "weights.pth"])   # can also load values from a list of str
print(cfg.dump())  # print formatted configs
with open("output.yaml", "w") as f:
  f.write(cfg.dump())   # save config to file
```

除了基本的 YAML 语法，配置文件还可以定义 `_BASE_: base.yaml` 字段，它将首先加载基本配置文件。如果存在任何冲突，基本配置中的值将在子配置中被覆盖。标准模型体系结构提供了几个基本配置。

detectron2 中的许多内置工具接受命令行配置覆盖: 命令行中提供的键-值对将覆盖配置文件中的现有值。例如，`demo.py` 可以

```bash
./demo.py --config-file config.yaml [--other-options] \
  --opts MODEL.WEIGHTS /path/to/weights INPUT.MIN_SIZE_TEST 1000
```

## 项目中的配置

存在于 detectron2 库之外的项目可以定义自己的配置，需要添加这些配置才能使项目正常运行，例如:

```bash
from detectron2.projects.point_rend import add_pointrend_config
cfg = get_cfg()    # obtain detectron2's default config
add_pointrend_config(cfg)  # add pointrend's default config
# ... ...
```

## 配置的最佳实践

1. 把你写的配置当成“代码”: 避免复制它们;使用 `_BASE_` 共享配置之间的公共部分。
2. 保持您编写的配置简单: 不要包括不影响实验设置的键。
