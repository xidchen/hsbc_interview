import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 0.2 * x ** 2 + 0.8 * np.log(1 / x ** 3) - 3 * x + 2


def df(x):
    return 0.4 * x - 2.4 / x - 3


def grad_descent(x, lr):
    return x - lr * df(x)


def main():
    x = np.linspace(0.01, 20, 100)
    y = f(x)
    plt.plot(x, y)
    plt.show()

    x = 1.5
    lr = 0.1
    for i in range(1000):
        x = grad_descent(x, lr)
        print(x)


if __name__ == '__main__':
    main()
