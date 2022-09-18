# import matplotlib.pyplot as plt
from numpy.random import randn
from datetime import datetime as dt
import numpy as np

data = {i: randn() for i in range(7)}
# print(data)

# plt.plot(randn(50).cumsum())
# plt.show()

a = str()

# print(isinstance(a, (int, float, bool)))


# check if an object is iterable (has the __iter__ method)
def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


# print(is_iterable(a), is_iterable(["a", "b", "c"]), is_iterable(int()))


# custom function that answers the question "object contains attribute"
def has_attribute(obj, attribute):
    try:
        getattr(obj, attribute)
        return True
    except AttributeError:
        return False


class Testing:
    name = "Test"


my_test = Testing()
# print(has_attribute(my_test, "name"))

string = r"\ \n \t \b"  # r'', `raw string` disregards special characters


# help(dt)
# datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])

# adding tuples
tup = (0, 2, 4), (3, 5)

tup += (["longer"], ["tuple"])
# print(tup)

# multiplying tuples
tup *= 2
# print(tup)


# NDARRAY (page 100)

# randn generates an x by y ndarray with x number of containers and y number of elements in each container.
# This is the 'shape' of the ndarray and it is an attribute (ndarray.shape).
data_2 = np.random.randn(2, 3)

# add to all values in the ndarray
data_2 + 10

# multiply all values in the ndarray by a constant
data_2 * 10

# add an ndarray to another ndarray (each corresponding cell)
data_2 + data_2

"""
An ndarray is a generic multidimensional container for homogeneous data; that is, all
of the elements must be the same type. Every array has a shape, a tuple indicating the
size of each dimension, and a dtype, an object describing the data type of the array.
"""

# print the shape of the ndarray
the_shape = data_2.shape

# print the data type of the ndarray
the_data_type = data_2.dtype

# Creating an ndarray using the np.array() function. Accepts sequence-like objects.
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

type(arr1)  # <class 'numpy.ndarray'>
data_type1 = arr1.dtype  # <class 'numpy.float64'>

list(arr1)

# nested sequences will be transformed into a multidimensional array
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)

data_type2 = arr2.dtype  # <class 'numpy.int32'>

ndim2 = arr2.ndim  # 2
shape2 = arr2.shape  # (2, 4)

print(arr2.dtype)
