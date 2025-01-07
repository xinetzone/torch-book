from torch import fx, nn

@fx.wrap
def wrapped_gemm_bias_mul(a, b, bias):
    lin_res = nn.functional.linear(a, b, bias=bias)
    mul_res = lin_res * a
    return lin_res, mul_res

@fx.wrap
def wrapped_gemm_bias_mul_with_c(a, b, bias, c):
    lin_res = nn.functional.linear(a, b, bias=bias)
    mul_res = lin_res * c
    return lin_res, mul_res