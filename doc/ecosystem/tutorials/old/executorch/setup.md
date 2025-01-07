# 配置 ExecuTorch

1. 初始化项目
    ```bash
    git clone https://github.com/pytorch/executorch.git
    cd executorch
    git submodule sync
    git submodule update --init
    ```
2. 创建环境(Python3.10)
    ```bash
    python3 -m venv .executorch
    source .executorch/bin/activate
    conda install cmake
    ./install_requirements.sh
    ```
3. 暴露 FlatBuffers 编译器：
    ExecuTorch 使用 flatc 来导出模型，并在第 `third-party/flatbuffers` 的源代码中构建它。通过将其添加到 `$PATH` 环境变量中使其可用，如上一步提示的那样，或者将其导出为 `$FLATC_EXECUTABLE` 环境变量。运行 `bash ./build/install_flatc.sh` 以确保正确安装了 `flatc`。
    ```bash
    export PATH="项目根目录/third-party/flatbuffers/cmake-out:${PATH}"
    bash 项目根目录/build/install_flatc.sh
    ```

## 测试 ExecuTorch 配置

在设置好环境后，您就可以将程序转换为 ExecuTorch 程序了。您需要使用 {func}`torch.export` 和 {mod}`executorch.exir` 来导出您的程序。然后，您可以将程序保存为 `.pte` 文件，这是 ExecuTorch 期望的文件扩展名。为了演示如何进行操作，我们将从 `nn.Module` 中生成 ExecuTorch 程序文件。

您可以通过使用示例脚本或 Python 解释器来生成 ExecuTorch 程序。

比如：

````{dropdown} 示例
```bash
source .executorch/bin/activate
export PATH="/media/pc/data/lxw/ai/executorch/third-party/flatbuffers/cmake-out:${PATH}"
bash /media/pc/data/lxw/ai/executorch/build/install_flatc.sh
python3 -m examples.portable.scripts.export --model_name="add"
```

```
[INFO 2023-10-25 13:44:47,751 utils.py:35] Core ATen graph:
graph():
    %arg0_1 : [num_users=3] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %arg0_1), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %arg0_1), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %add_2), kwargs = {})
    return (add_3,)
[INFO 2023-10-25 13:44:47,836 utils.py:50] Exported graph:
graph():
    %arg0_1 : [num_users=3] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
    %aten_add_tensor_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor, %arg0_1), kwargs = {})
    %aten_add_tensor_2 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_1, %arg0_1), kwargs = {})
    %aten_add_tensor_3 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_2, %aten_add_tensor_2), kwargs = {})
    return (aten_add_tensor_3,)
[INFO 2023-10-25 13:44:48,001 utils.py:86] Saved exported program to add.pte
```

这个命令已经创建了名为 `add.pte` 的文件，其中包含了您的示例程序。

或者，您可以使用 Python 解释器执行相同的操作：

```python
>>> import executorch.exir as exir
>>> from executorch.exir.tests.models import Mul
>>> m = Mul()
>>> print(exir.capture(m, m.get_random_inputs()).to_edge())
>>> open("mul.pte", "wb").write(exir.capture(m, m.get_random_inputs()).to_edge().to_executorch().buffer)
```
````

在这一步，您学习了如何将您的 PyTorch 程序导出为 ExecuTorch 程序。您可以将相同的原理应用于自己的 PyTorch 程序。

下一步是设置 Buck2 并构建 executor_runner 来运行您的程序。

## 构建 executor 运行时

导出程序后，您几乎已经准备好运行它了。下一步涉及使用 Buck2 构建运行时。

Buck2 是开源的构建系统，使开发人员能够轻松高效地管理项目依赖项。我们将使用 Buck2 来构建 `executor_runner`，这是 ExecuTorch 运行时的示例包装器，其中包含所有算子和后端。

需要以下前提条件：

1. 安装 `zstd`
    ```bash
    pip3 install zstd
    ```
2. Buck2 命令行工具的 2023-07-18 版本 - 您可以从 [Buck2 存储库](https://github.com/facebook/buck2/releases/tag/2023-07-18)中下载适用于您系统的预构建存档。请注意，版本很重要，较新的或较旧的版本可能无法与 ExecuTorch 存储库使用的 Buck2 预置版本一起使用。
3. 通过以下命令解压缩来配置 Buck2（文件名取决于您的系统）：
```
# For example, buck2-x86_64-unknown-linux-musl.zst or buck2-aarch64-apple-darwin.zst
zstd -cdq buck2-DOWNLOADED_FILENAME.zst > /tmp/buck2 && chmod +x /tmp/buck2
```

您可能需要将 `buck2` 二进制文件复制到 `$PATH` 中，以便可以作为 `buck2` 运行它。
```bash
/tmp/buck2 build //examples/portable/executor_runner:executor_runner --show-output
```

## 运行 executor 程序

```bash
/tmp/buck2 run //examples/portable/executor_runner:executor_runner -- --model_path add.pte
```

或者

```bash
./buck-out/.../executor_runner --model_path add.pte
```
