# import matplotlib.pyplot as plt
from numpy.random import randn
from datetime import datetime as dt
import numpy as np

random_data = {i: randn() for i in range(7)}

# plt.plot(randn(50).cumsum())
# plt.show()

a_string = str()
# print(isinstance(a_string, (int, float, bool)))


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
has_attribute(my_test, "name")

raw_string = r"\ \n \t \b"  # r'', `raw string` disregards special characters

# help(dt)
# datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])

# adding tuples
tup = (0, 2, 4), (3, 5)

tup += (["longer"], ["tuple"])

# multiplying tuples
tup *= 2


# NDARRAY INTRODUCTION (page 100)

"""
An ndarray is a generic multidimensional container for homogeneous data; that is, all
of the elements must be the same type. Every array has a shape, a tuple indicating the
size of each dimension, and a dtype, an object describing the data type of the array (elements).
"""

# dtypes can be written in 'short code' (f16, u4, etc)

# randn generates an x by y ndarray with x number of containers and y number of elements in each container.
# This is the 'shape' of the ndarray and it is an attribute (ndarray.shape).
data_2 = np.random.randn(2, 3)

# add to all values in the ndarray
data_2 + 10

# multiply all values in the ndarray by a constant
data_2 * 10

# add an ndarray to another ndarray (each corresponding cell)
data_2 + data_2

# print the shape of the ndarray
the_shape = data_2.shape

# print the data type of the ndarray
the_data_type = data_2.dtype

# Creating ndarrays

# 1. Uing the np.array() function. Accepts sequence-like objects. (page 101)
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

# 2. Using np.zeros(). Accepts the shape tuple as a parameter. Has a "zeros_like" method.
# If only one value is given, the tuple is assumed to be (1, x) and requires no extra parentheses.
data_zeros = np.zeros(5)

# 3. Using np.empty(). Accepts the shape tuple as a parameter. Has a "empty_like" method.
data_empty = np.empty((1, 5))
data_empty_like = np.empty_like(data_zeros)

# 4. Using np.arange() 'array range'. Accepts scalar instead of tuple as a parameter.
data_arange = np.arange(start=0, stop=24, step=2)

# 5. Using np.ones(). Accepts the shape tuple as a parameter. Has a "ones_like" method.
data_ones = np.ones((2, 6))
data_ones_like = np.ones_like(data_empty)

# other np methods (page 103)
# test = np.asarray(input("Input the data and press Enter"))

# if M is not specified, defaults to M = N
matrix_eye = np.eye(N=3, dtype=np.int8)
# >>> [[1 0 0] [0 1 0] [0 0 1]]

matrix_identity = np.identity(3, dtype=np.float64)

# numpy data types (page 104)

# Changing the dtype of an ndarray

# 1. From int to float
float_data2 = np.array(data2, dtype=np.int8).astype(np.float64)

# 2. From any type to the type of target other array (using the ndarray.dtype attribute)
my_data = np.array(["-.96", "1.876", "-1.008", "1.581"], dtype=np.string_)
my_data = my_data.astype(float_data2.dtype)

# CAUTION! np.string_ data type has a fixed size and will truncate without warning.

"""

_______________________________________________________________________________
|  Data Type   |       Output Range                                            |
|______________________________________________________________________________|
|  int8        |       -128 to 127                                             |
|______________________________________________________________________________|
|  int16       |       -32,768 to 32,767                                       |
|______________________________________________________________________________|
|  int32       |       -2,147,483,648 to 2,147,483,647                         |
|______________________________________________________________________________|
|  int64       |       -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 |
_______________________________________________________________________________|


"""

from sys import getsizeof as size

print("\nnp.float64 takes up more space than python float:", size(np.float64(2)) > size(float(2)))
print("np.float16 takes up more space than python float:", size(np.float16(2)) > size(float(2)))
print("np.int64 takes up more space than python int:", size(np.int64(2)) > size(int(2)))
print("np.int8 takes up more space than python int:", size(np.int8(2)) > size(int(2)))
