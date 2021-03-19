import warnings
from functools import reduce
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')