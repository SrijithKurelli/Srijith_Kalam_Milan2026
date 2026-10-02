"""Small local helper; the official evaluator calls Model.train directly."""
from .model import Model


def fit_model(X, y):
    m = Model()
    m.train(X, y)
    return m
