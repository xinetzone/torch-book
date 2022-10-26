import torch

def linear(X, X0, X1, Y0, Y1):
    # scale = (Y1 - Y0)/(X1 - X0)
    # zero_point = Y0 - scale * X0
    # f = x * scale + zero_point
    l0 = (X-X1)/(X0 - X1)
    l1 = (X-X0)/(X1 - X0)
    return l0 * Y0 + l1 * Y1

def get_nodes(features, n_samples=10000, bins=10):
    # X = torch.randn(n_samples, 16, 4, 4)
    full_index = torch.sigmoid(features).argsort(dim=0)
    index = full_index[::n_samples//bins]
    return torch.concat([features.take(index), 
                         features.take(full_index[-1]).unsqueeze(dim=0)])

def _sigmoid2linear(X, n_samples=10000, bins=10):
    Xs = get_nodes(X, n_samples=n_samples, bins=bins)
    bins = len(Xs) - 1
    Ys = torch.sigmoid(Xs)
    Y = torch.zeros_like(X)
    for k in range(bins):
        interval = (X >= Xs[k]) * (X <= Xs[k+1])
        Y += linear(X, Xs[k], Xs[k+1], Ys[k], Ys[k+1]) # interval
    return Y

def sigmoid2linear(Xs, n_samples=10000, bins=10):
    return torch.stack([_sigmoid2linear(x, 
                                        n_samples=n_samples, 
                                        bins=bins) 
                        for x in Xs])