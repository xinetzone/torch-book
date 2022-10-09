from torch import jit, randn
from tvm.ir.transform import PassContext
from tvm import relay, device
from tvm.contrib.graph_executor import GraphModule


def create_relay_model(model, input_shape, name):
    # Get off the shelf model, and convert to relay
    input_data = randn(input_shape)
    # model = getattr(models, model_name)(pretrained=True).eval().cpu()
    scripted_model = jit.trace(model, input_data).eval()
    #转成relay格式，输入层名字可任意指定
    shape_list = [(name, input_shape)]
    # mod, params =
    return relay.frontend.from_pytorch(scripted_model, shape_list)


def build_tvm_lib(mod, params,
                  target="llvm",
                  mod_name="default"):
    """
    Args:
        data: Numpy array
    
    Examples:
        >>> lib = build_lib(...)
        >>> dev = tvm.device(target, device_id)
        >>> runtime = GraphModule(lib["default"](dev))
        >>> res = inference(runtime, data, name)
    """
    # model_name = 'alexnet'
    # target = "llvm"
    with PassContext(opt_level=3):
        # model = getattr(models, model_name)(pretrained=True).eval().cpu()
        # model = model.features
        # mod = quantize(mod, params, mode,
        #                dataset=dataset,
        #                skip_conv_layers=skip_conv_layers,
        #                skip_dense_layer=skip_dense_layer)

        lib = relay.build(mod,
                          target=target,
                          params=params,
                          mod_name=mod_name)
    return lib


def graph_module(lib,
                 target="llvm",
                 device_id=0,
                 mod_name="default"):
    dev = device(target, device_id)
    return GraphModule(lib[mod_name](dev))
