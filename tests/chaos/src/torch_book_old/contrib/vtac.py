import json
from vta import get_env, build_config
from vta.top import graph_pack
from tvm.ir.transform import PassContext
from tvm import autotvm, relay, IRModule


def update_vta_config(tvm_root, **kw):
    vat_config = f"{tvm_root}/3rdparty/vta-hw/config/vta_config.json"
    with open(vat_config, encoding="utf-8") as fp:
        config = json.load(fp)

    config.update(kw)
    with open(vat_config, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=4)


def build_vta_lib(mod, params, target, tune_config, tvm_root):
    """
    Args:
        target: "xmfpga", "sim"
    """
    update_vta_config(tvm_root, **{'TARGET': target})
    env = get_env()
    with PassContext(opt_level=3):
        # mod = quantize(mod, params, mode,
        #                dataset=dataset,
        #                skip_conv_layers=skip_conv_layers,
        #                skip_dense_layer=skip_dense_layer)
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH
        )

        # 得到模型最优配置
        mod = IRModule.from_expr(relay_prog)
        tasks = autotvm.task.extract_from_program(
            mod,
            params=params,
            ops=(relay.op.get("nn.conv2d"),
                 relay.op.get("nn.dense")),
            target=env.target
        )
        # logging.info(f'tasks: {tasks}')
        tune_path = tune_config.get("tune_path", "")
        if not tune_path:
            tune_path = "tune/res.log"
            autotvm.utils.find_config(tasks, tune_path)
        # 编译(交叉编译,输出动态库)
        # 输出指令 debug_flag=2
        with autotvm.tophub.context(env.target, [tune_path]):
            with build_config(opt_level=3,
                              disable_vectorize=False,
                              disabled_pass={"AlterOpLayout"}):
                # Relay先会寻找 AutoTVM 是否有预先tune好的参数记录
                lib = relay.build(relay_prog,
                                  target=env.target,
                                  params=params,
                                  target_host=env.target_host)
    return lib


def save_vta_lib(lib, target, lib_path="model.so"):
    if "xmfpga" in target:
        lib.export_library(lib_path,
                           cc="arm-linux-gnueabihf-gcc")
    else:
        lib.export_library(lib_path)
