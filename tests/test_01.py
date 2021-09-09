import numpy as np

def f(x1, x2, x3, x4, x5, x6):
    e1 = x1-1
    e2 = x2-1
    e3 = x3-1
    e4 = x4-1
    e5 = x5-1
    e6 = x6-1
    return np.array([e1,e2,e3,e4,e5,e6])

list = np.array([ [x1, x2, x3, x4, x5, x6] for x1 in range(-1,2) for x2 in range(-1,2) for x3 in range(-1,2)
                                    for x4 in range(-1,2) for x5 in range(-1,2) for x6 in range(-1,2) ])
print(list)