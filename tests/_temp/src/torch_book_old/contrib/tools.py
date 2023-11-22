import logging
from tvm.ir.transform import PassContext
from tvm import relay


def calibrate_power2(mod, params, dataset,
                     skip_conv_layers=[],
                     skip_dense_layer=False,
                     **kwargs):
    logging.debug(f"量化前: \n{mod}")
    kwargs.update({
        "skip_conv_layers": skip_conv_layers,
        "skip_dense_layer": skip_dense_layer,
        "activation_scale_power2": True
    })
    logging.debug(f"量化前: \n{mod}")
    with PassContext(opt_level=3):
        with relay.quantize.qconfig(calibrate_mode="kl_divergence",
                                    weight_scale="power2",
                                    **kwargs):
            mod = relay.quantize.quantize(mod, params, dataset)
            current_qconfig = relay.quantize.current_qconfig()
            logging.info(f"当前 qconfig: {current_qconfig}")
            logging.debug(f"量化后(power2): \n{mod}")
    return mod


def calibrate_max(mod, params, dataset,
                  skip_conv_layers=[],
                  skip_dense_layer=False,
                  **kwargs):
    logging.debug(f"量化前: \n{mod}")
    kwargs.update({
        "skip_conv_layers": skip_conv_layers,
        "skip_dense_layer": skip_dense_layer
    })
    with PassContext(opt_level=3):
        with relay.quantize.qconfig(calibrate_mode="kl_divergence",
                                    weight_scale="max",
                                    **kwargs):
            mod = relay.quantize.quantize(mod, params, dataset)
            current_qconfig = relay.quantize.current_qconfig()
            logging.info(f"当前 qconfig: {current_qconfig}")
            logging.debug(f"量化后(max): \n{mod}")
    return mod


def global_quantize(mod, params,
                    skip_conv_layers=[],
                    skip_dense_layer=False,
                    **kwargs):
    logging.debug(f"量化前: \n{mod}")
    kwargs.update({
        "skip_conv_layers": skip_conv_layers,
        "skip_dense_layer": skip_dense_layer
    })
    with PassContext(opt_level=3):
        with relay.quantize.qconfig(calibrate_mode="global_scale",
                                    global_scale=8.0,
                                    **kwargs):
            mod = relay.quantize.quantize(mod, params)
            current_qconfig = relay.quantize.current_qconfig()
            logging.info(f"当前 qconfig: {current_qconfig}")
            logging.debug(f"量化后(global): \n{mod}")
    return mod


def save_info(lib, output_name):
    with open(f"{output_name}.json", 'w') as f:
        f.write(lib.graph_json)
        # dump_ops(json.loads(lib.graph_json), output_name, 'log')
    with open(f"{output_name}.params", 'wb') as f:
        f.write(relay.save_param_dict(lib.params))


def inference(runtime, data, name='data'):
    runtime.set_input(name, data)
    runtime.run()
    output = runtime.get_output(0)
    return output #.asnumpy()

# def save_result(lib,
#                 target,
#                 name,
#                 x,
#                 save_path):
#     import pandas as pd
#     ctx = device(target, 0)
#     module = GraphModule(lib["default"](ctx))
#     tvm_output = inference(module, x, name)
#     output = pd.DataFrame(tvm_output)
#     output.to_csv(save_path, index=False)