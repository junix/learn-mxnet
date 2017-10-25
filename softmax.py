from math import exp


def softmax(xs):
    xs = [exp(float(x)) for x in xs]
    sm = sum(xs)
    for x in xs:
        yield x / sm


if __name__ == '__main__':
    cs = range(10)
    for a, b in zip(cs, softmax(cs)):
        print('{} -> {}'.format(a, b))
