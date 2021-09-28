import numpy as np

from typing import Iterable


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_delta(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1.0 - sigmoid(x))


def sum_sq_err(x: np.ndarray, y: np.ndarray) -> float:
    return np.sum((x - y) ** 2)


def sum_sq_err_delta(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * (x - y)


class NeuralNetwork:
    def __init__(self, seed: int, dims: list[int], learning_rate: float):
        rs = np.random.RandomState(seed)
        self.lr = learning_rate
        self.cs = [rs.rand(o, i + 1) for i, o in zip(dims, dims[1:])]
        self.N = len(self.cs)

    def run(self, i: np.ndarray) -> np.ndarray:
        self.z = [self.cs[0] @ np.append(i, 1.0)]
        self.a = [sigmoid(self.z[0])]

        for c in self.cs[1:]:
            self.z.append(c @ np.append(self.a[-1], 1.0))
            self.a.append(sigmoid(self.z[-1]))

        return self.a[-1]

    def gradients(self, i: np.ndarray, o: np.ndarray) -> list[np.ndarray]:
        a = self.run(i)
        self.d: list[np.ndarray] = [None] * self.N  # type: ignore
        self.d[self.N - 1] = sigmoid_delta(self.z[-1]) * sum_sq_err_delta(a, o)

        for n in range(self.N - 2, -1, -1):
            self.d[n] = sigmoid_delta(self.z[n]) * (
                self.cs[n + 1][:, :-1].T @ self.d[n + 1]
            )

        self.g = [
            np.outer(self.d[n], np.append(self.a[n - 1] if n > 0 else i, 1.0))
            for n in range(self.N)
        ]
        return self.g

    def fit_one(self, i: np.ndarray, o: np.ndarray) -> None:
        g = self.gradients(i, o)
        for n in range(self.N):
            self.cs[n] -= g[n] * self.lr

    def train(self, data: Iterable[tuple[np.ndarray, np.ndarray]]) -> None:
        self.loss = []
        for n, (i, o) in enumerate(data):
            self.fit_one(i, o)
            self.loss.append(sum_sq_err(self.a[-1], o))

    def test(self, data: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[float]:
        return [sum_sq_err(self.run(i), o) for i, o in data]

