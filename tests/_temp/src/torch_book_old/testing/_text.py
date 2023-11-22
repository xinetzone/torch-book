import torch
from backend.gpu import try_gpu

def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence.

    Defined in :numref:`sec_sentiment_rnn`"""
    sequence = torch.tensor(vocab[sequence.split()], device=try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
