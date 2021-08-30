import numpy as np
import tensorflow as tf

def normalize(input):
    input = np.float32(input)
    xmin = np.amin(input)
    xmax = np.amax(input)
    b = 1.  # max value
    a = 0.  # min value
    if (xmax - xmin) == 0:
        out = input
    else:
        out = a+(b-a)*(input-xmin)/(xmax-xmin)
    return out