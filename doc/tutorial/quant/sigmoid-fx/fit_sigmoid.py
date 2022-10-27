import torch

def linear(X, X0, X1, Y0, Y1):
    # scale = (Y1 - Y0)/(X1 - X0)
    # zero_point = Y0 - scale * X0
    # f = x * scale + zero_point
    l0 = (X-X1)/(X0 - X1)
    l1 = (X-X0)/(X1 - X0)
    return l0 * Y0 + l1 * Y1

def get_nodes(features, bins=10):
    # X = torch.randn(n_samples, 16, 4, 4)
    n_samples = len(features)
    step = n_samples//bins
    assert step > 0, "校准数据不匹配"
    full_index = features.argsort(dim=0)
    index = full_index[::step]
    index = torch.concat([full_index[::step],
                          full_index[-1].unsqueeze(dim=0)])
    return features.take(index)

def _sigmoid2linear(X, bins=10):
    Xs = get_nodes(X, bins=bins)
    Ys = torch.sigmoid(Xs)
    Y = torch.zeros_like(X) * (X < -6)
    for k in range(bins):
        interval = (X >= Xs[k]) * (X <= Xs[k+1]) + (X <= Xs[k]) * (X >= Xs[k+1])
        Y += linear(X, Xs[k], Xs[k+1], Ys[k], Ys[k+1]) * interval
    return Y

def sigmoid2linear(Xs, bins=10):
    return torch.stack([_sigmoid2linear(x, bins=bins) 
                        for x in Xs])