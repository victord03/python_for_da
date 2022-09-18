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

# 1. Uing the np.array() function. Accepts sequence-like objects or a tuple. (page 101)
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

"""
MEMORY CONSUMPTION

from sys import getsizeof as size

print("\nnp.float64 takes up more space than python float:", size(np.float64(2)) > size(float(2)))
print("np.float16 takes up more space than python float:", size(np.float16(2)) > size(float(2)))
print("np.int64 takes up more space than python int:", size(np.int64(2)) > size(int(2)))
print("np.int8 takes up more space than python int:", size(np.int8(2)) > size(int(2)))"""

# VECTORIZATION is to apply batch operations on data without loops (page 106).

# Any arithmetic operations between equal-size arrays applies the operation element-wise (page 106).
numbers = np.array(([2, 3, 5, 7], [10, 20, 50, 100]))

numbers * numbers
numbers + numbers

# Arithmetic operations with scalars propagate the scalar argument to each element in the array (page 106).
1 / numbers
numbers + 5

# Comparisons between arrays of the same size yield boolean arrays (page 107).
numbers_2 = np.array(([9, 11, 13, 17], [19, 23, 53, 113]))
comparison = numbers > (numbers_2 - 8)

# Operations between differently sized arrays is called broadcasting and will be discussed later.

# BASIC INDEXING AND SLICING (page 107).
# CAUTION! Slicing [a:b] includes a, but excludes b (stops at b-1)

# 1. One-dimensional arrays are simple; on the surface they act similarly to Python lists
numbers_to_fifteen = np.arange(1, 16)

# CAUTION! Slicing in arrays are actually "views" on the original arrays. Any modifications will be reflected
# in the origin.
view_1 = numbers_to_fifteen[1:3]
view_1[:] = 404  # will be assigned to all values in the current array view

# To copy the slice to a new object (therefore avoiding modifying the origin) an explicit copy must be made.
my_slice_copy = numbers_to_fifteen[0:9].copy()
my_slice_copy[:] = 99

# 2. Multi-dimensional arrays can be accessed either recursively or by a comma-separated list of indices (page 108).
assert numbers_2[0][3] == numbers_2[0, 3]  # does not raise AssertionError

new_ndarray = np.ones((3, 4, 2), dtype=np.int8)

"""

VALUABLE CHEAT SHEET (INDEXING, SLICING)

2 x 2 "Two containers, each one of those containers containing two elements inside."

[
    [  { [0] }, { [1] }  ],                      < index [0]
    [  { [0] }, { [1] }  ],                      < index [1]
]

3 x 3 "Three containers, each one of those containers containing three elements inside."

[
    [  { [0] }, { [1] }, { [2] }  ],             < index [0]
    [  { [0] }, { [1] }, { [2] }  ],             < index [1]
    [  { [0] }, { [1] }, { [2] }  ],             < index [2]
]

2 x 2 x 3 "Two layers, of two containers, each one of those containers containing three elements inside."

[
   [                                            < index [0]
        [  { [0] }, { [1] }, { [2] }  ],                     < index [0][0]
        [  { [0] }, { [1] }, { [2] }  ],                     < index [0][1]
   ],                                       
   [                                            < index [1]
        [  { [0] }, { [1] }, { [2] }  ],                     < index [1][0]
        [  { [0] }, { [1] }, { [2] }  ],                     < index [1][1]
   ],
]

new_ndarray[:, :1]  # selects the containers in the slice [:1], from each ([:]) layer.
new_ndarray[:, :1, 2]  # selects the value at index 2, from the containers in the slide [:1], from each ([:]) layer.

add several comma-separated slices, to dive into the different dimensions

      layer    container    value
        v          v          v
  [   0:-1,      0:-1,      0:-1   ]


3 x 4 x 2 "Three layers, of four containers, each one of those containers containing two elements inside."

[
    [                                           < index [0]
        [  { [0] }, { [1] }  ],                              < index [0][0]
        [  { [0] }, { [1] }  ],                              < index [0][1]
        [  { [0] }, { [1] }  ],                              < index [0][2]
        [  { [0] }, { [1] }  ],                              < index [0][3]
    ],
    [                                           < index [1]
        [  { [0] }, { [1] }  ],                              < index [1][0]
        [  { [0] }, { [1] }  ],                              < index [1][1]
        [  { [0] }, { [1] }  ],                              < index [1][2]
        [  { [0] }, { [1] }  ],                              < index [1][3]
    ],
    [                                           < index [2]
        [  { [0] }, { [1] }  ],                              < index [2][0]
        [  { [0] }, { [1] }  ],                              < index [2][1]
        [  { [0] }, { [1] }  ],                              < index [2][2]
        [  { [0] }, { [1] }  ],                              < index [2][3]
    ],
]

code example 1 (update the values from each of the two middle rows, for each container and layer, to zeros): 
new_ndarray = np.ones((3, 4, 2), dtype=np.int8)
new_ndarray[:, 1:3] = 0
new_ndarray[:, 1:3].shape >>> (3, 2, 2)

code example 2 (update the last value on each row, for each container and layer, to zeros):
new_ndarray = np.ones((3, 4, 2), dtype=np.int8)
new_ndarray[:, :, 1] = 0
new_ndarray[:, 1:3].shape >>> (3, 4) (Notice that the trailing 1 is omitted (3, 4, 1)).

More examples in section 'Two-dimensional array slicing' (page 112).
"""

# BOOLEAN INDEXING (page 112).
"""
Using the slice operator on an array with an array of boolean values, 
will yield all containers at the corresponding indices where True was found / matched.

The boolean array should be of the same length as the array it's indexing (and it will not fail if it isn't).

It is also possible to mix and match boolean arrays with slices or integers. Examples below.
"""
names = np.array(["Bob", "Alfred", "Bob", "Joyce", "Will", "Joyce", "Joe"])
random_data_2 = np.random.randn(7, 4)

print(random_data_2)
print("\n\n")
boolean_array_bob = names == "Bob"
selection = random_data_2[boolean_array_bob]  # excuse me. wat.

selection_2 = random_data_2[boolean_array_bob, 1:3]

# to inverse selection it is possible to either negate the operator '!= "Bob"', or negate the condition using ~
selection_3 = random_data_2[~boolean_array_bob, -1]

# It is possible to use | (or) and & (and) operators for boolean statements. Use parenthesis for each statement.
boolean_array_bob_or_joyce = (names == "Bob") | (names == "Joyce")
list_of_indices = [index for index, boolean in enumerate(list(boolean_array_bob_or_joyce)) if boolean]
selection_bob_or_joyce = random_data_2[boolean_array_bob_or_joyce]
selection_not_bob_nor_joyce = random_data_2[~boolean_array_bob_or_joyce]
