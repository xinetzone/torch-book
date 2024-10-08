import numpy as np
from tabulate import tabulate

def flops_table(gm, sample_input):
    result_table = []
    for node in gm.graph.nodes:
        node: Node
        _result_row = [node.name, node.op, str(node.target)]

        node_module_name = ''
        if (_var_name := 'nn_module_stack') in node.meta:
            node_module_name = next(reversed(node.meta[_var_name].values())).__name__
            # print(node_module_name)
            # node_module_name = ".".join([_v.__name__ for _v in node.meta[_var_name].values()])
        _result_row.append(node_module_name)

        if (_var_name := 'FLOPs') in node.meta:
            flops = node.meta[_var_name]
            if flops is None:
                _result_row.append('not_recognized')
            elif isinstance(flops, int|np.int64):
                _result_row.append(flops)
            else:
                raise TypeError(type(flops))
        else:
            raise KeyError("'FLOPs' must be in node.meta")

        result_table.append(_result_row)
    return result_table

def show_flops_table(gm, sample_input):
    result_header = ['node_name', 'node_op', 'op_target', 'nn_module_stack[-1]', 'FLOPs']
    result_table = flops_table(gm, sample_input)
    __missing_values = [''] * 4 + ['ERROR']
    # table_str = tabulate(result_table, result_header, tablefmt='rst', missingval=__missing_values)
    table_str = tabulate(result_table, result_header, tablefmt='fancy_grid', missingval=__missing_values)
    print(table_str)
    valid_flops_list = list(filter(lambda _f: isinstance(_f, int|np.int64), list(zip(*result_table))[-1]))
    total_flops = sum(valid_flops_list)
    num_empty_flops = len(result_table) - len(valid_flops_list)
    print(f"total_flops = {total_flops:3,}")