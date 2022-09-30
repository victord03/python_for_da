# import matplotlib.pyplot as plt
import random

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

# check number of dimensions in the array
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

3 x 1 x 4

[
    [  { [0] }, { [1] }, { [2] }, { [3] }  ],     < index [0]
    [  { [0] }, { [1] }, { [2] }, { [3] }  ],     < index [1]
    [  { [0] }, { [1] }, { [2] }, { [3] }  ],     < index [2]
]

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

boolean_array_bob = names == "Bob"
selection = random_data_2[boolean_array_bob]  # excuse me. wat.

selection_2 = random_data_2[boolean_array_bob, 1:3]

# to inverse selection it is possible to either negate the operator '!= "Bob"', or negate the condition using ~
selection_3 = random_data_2[~boolean_array_bob, -1]

# It is possible to use | (or) and & (and) operators for boolean statements. Use parenthesis for each statement.
boolean_array_bob_or_joyce = (names == "Bob") | (names == "Joyce")
list_of_indices = [
    index for index, boolean in enumerate(list(boolean_array_bob_or_joyce)) if boolean
]
selection_bob_or_joyce = random_data_2[boolean_array_bob_or_joyce]
selection_not_bob_nor_joyce = random_data_2[~boolean_array_bob_or_joyce]

# FANCY INDEXING (page 115)
arr3 = np.empty((8, 4))

for i in range(8):
    arr3[i] = i

# print(arr3)
# print("\n\n")

# Passing a list of indices as a index, will return the corresponding containers at the specified indices
# Handles negative indices as in standard Python.
specific_rows = arr3[[0, 3, 5, 7]]

arr4 = np.arange(32).reshape((8, 4))  # reshape will be explored later in the book.

# Passing a tuple of lists of indices returns, for each container at index on the first list, the element at index
# specified in the second list.
selection_4 = arr4[
    [0, 2, 5], [1, 2, 0]
]  # This returns the positions (0, 1), (2, 2) and (5, 0) from the ndarray

selection_5 = arr4[[0, 2, 5]]
selection_6 = arr4[[0, 2, 5]][
    :, [3, 2]
]  # This returns the rows [0, 2, 5] and then the columns 3 and 2

# TRANSPOSING ARRAYS AND SWAPPING AXES (page 116)
arr5 = np.arange(15).reshape((5, 3))

# the special .T attribute flips the array horizontally or vertically, maintaining the order on which the elements
# appeared in. See the quote block for more info on the ordering.

"""

3 x 5 shape(3, 5)

OVERVIEW

    Initial state (arr):
        [0  1  2  3  4]
        [5  6  7  8  9]
        [10 11 12 13 14]
    
    Transposed state (arr.T): 
        [0  5 10]
        [1  6 11]
        [2  7 12]
        [3  8 13]
        [4  9 14]

INDICES ANALYSIS

    Initial state (arr):
        [  [0][0]  [0][1]  [0][2]  [0][3]  [0][4]  ]
        [  [1][0]  [1][1]  [1][2]  [1][3]  [1][4]  ]
        [  [1][0]  [1][1]  [1][2]  [1][3]  [1][4]  ]
    
    Transposed state (arr.T): 
        [  [0][0]  [1][0]  [2][0]  ]
        [  [0][1]  [1][1]  [2][1]  ]
        [  [0][2]  [1][2]  [2][2]  ]
        [  [0][3]  [1][3]  [2][3]  ]
        [  [0][4]  [1][4]  [2][4]  ]


5 x 3 shape(5, 3)

OVERVIEW

    Initial state (arr):
        [ 0  1  2]
        [ 3  4  5]
        [ 6  7  8]
        [ 9 10 11]
        [12 13 14]

    Transposed state (arr.T): 
        [ 0  3  6  9 12]
        [ 1  4  7 10 13]
        [ 2  5  8 11 14]

INDICES ANALYSIS

    Initial state (arr):
        [  [0][0]  [0][1]  [0][2]  ]
        [  [1][0]  [1][1]  [1][2]  ]
        [  [2][0]  [2][1]  [2][2]  ]
        [  [3][0]  [3][1]  [3][2]  ]
        [  [4][0]  [4][1]  [4][2]  ]
        
    
    Transposed state (arr.T): 
        [  [0][0]  [1][0]  [2][0]  [3][0]  [4][0]  ]
        [  [0][1]  [1][1]  [2][1]  [3][1]  [4][1]  ]
        [  [0][2]  [1][2]  [2][2]  [3][2]  [4][2]  ]
"""

# UNIVERSAL FUNCTIONS: FAST ELEMENT-WISE ARRAY FUNCTIONS (page 118)

"""

np.minimum(), np.maximum()

arr6 = np.random.randn(4, 4)
arr7 = np.random.randn(4, 4)

print("\nArray 1")
print(abs(arr6))
print("\n\nArray 2")
print(abs(arr7))

print("\n\nMaximum")
print(np.maximum(abs(arr6), abs(arr7)))
print("\n\nMinimum")
print(np.minimum(abs(arr6), abs(arr7)))

Unary ufuncs (page 120)

Binary universal funcs (page 120)

.add                    Add corresponding elements in arrays
.subtract               Subtract elements in second array from first array
.multiply               Multiply array elements
.divide, floor_divide   Divide or oor divide (truncating the remainder)
.power                  Raise elements in first array to powers indicated in second array
.maximum, fmax          Element-wise maximum; fmax ignores NaN
.minimum, fmin          Element-wise minimum; fmin ignores NaN
.mod                    Element-wise modulus (remainder of division)
.copysign               Copy sign of values in second argument to values in first argument

.greater
.greater_equal
.less
.less_equal
equal
.not_equal              Perform element-wise comparison, yielding boolean array (equivalent to infix operators >, >=, 
<, <=, ==, !=)

.logical_and
.logical_or
.logical_xor            Compute element-wise truth value of logical operation (equivalent to infix operators & |, ^)

"""

# ARRAY-ORIENTED PROGRAMMING WITH ARRAYS (page 121)
# The np.meshgrid function takes two 1D arrays and produces two 2D matrices corresponding
# to all pairs of (x, y) in the two arrays

# EXPRESSING CONDITIONAL LOGIC AS ARRAY OPERATIONS (page 122)
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64)
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5], dtype=np.float64)
cond = np.array([True, False, True, True, False], dtype=np.bool_)

result = np.where(cond, xarr, yarr)

"""
Suppose you had a matrix of randomly generated data and you wanted to replace 
some of the values, based on a specific condition, using the boolean indexing.
"""
arr6 = np.random.randn(4, 4)
cond_arr = (-1.2 < arr6) & (arr6 < 1.2)  # Df = (-1.2, 1.2)
result_cond = np.where(
    cond_arr, 1, 0
)  # Replace all values between -1.2 and 1.2 with 1, and the rest with 0.

neg_to_zero = arr6 < 0
"\nNegatives to zero"
np.where(
    neg_to_zero, 0, arr6
)  # Replace all negative values with 0, leave the others intact.

# MATHEMATICAL AND STATISTICAL METHODS (page 124)
arr7 = np.array([np.arange(1, 17)], np.float16)
arr7 = arr7.reshape(4, 4)

# Populating with semi-random data manually (np.random.randn() could be used but slicing is practiced here).
arr7[[0, 0, 0, 2, 3], [0, 1, 2, 0, 2]] = -2.008
arr7[[0, 1, 1], [3, 2, 3]] = 2.48
arr7[[1, 2, 2, 2], [1, 0, 2, 3]] = -0.966
arr7[[1, 2, 3, 3, 3], [0, 1, 0, 1, 3]] = 1.114

print_mean = "\nMean (Overall)"
# Mean, sum, std (standard deviation). Can be called as methods or top-level NumPy functions.
mean_array_overall = (
    arr7.mean()
)  # optional argument axis=0 for the columns (vertical) or axis=1 for the rows (horizontal).
assert arr7.mean() == np.mean(arr7)
# print("_" * 36)

print_mean_1 = "\nMean (Horizontal)"
mean_array_hori = arr7.mean(axis=1)

"""
Could be accessed as a list

print("\nMean (row 0)")
print(arr7.mean(axis=1)[0])
print("\nMean (row 1)")
print(arr7.mean(axis=1)[1])
print("\nMean (row 2)")
print(arr7.mean(axis=1)[2])
print("\nMean (row 3)")
print(arr7.mean(axis=1)[3])"""

print_mean_2 = "\nMean (Vertical)"
mean_array_vert = arr7.mean(axis=0)

"""
Could be accessed as a list

print("\n\nMean (column 0)")
print(arr7.mean(axis=0)[0])
print("\nMean (column 1)")
print(arr7.mean(axis=0)[1])
print("\nMean (column 2)")
print(arr7.mean(axis=0)[2])
print("\nMean (column 3)")
print(arr7.mean(axis=0)[3])"""

print_sum = "\nSum (Overall)"
sum_array_overall = arr7.sum()
assert arr7.sum() == np.sum(arr7)
# print("_" * 36)

print_sum_1 = "\nSum (Horizontal)"
sum_array_hori = arr7.sum(axis=1)

print_sum_2 = "\nSum (Vertical)"
sum_array_vert = arr7.sum(axis=0)

# Cumulative sum (cumsum) returns an array with the cumulative sum for each number encountered so far,
# at each index.
arr8 = np.array([x for x in range(1, 9)])
cumulative_sum = arr8.cumsum()  # cumulative sum (up to each index)

# Cumulative product (cumprod) returns an array with the cumulative product.
arr9 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
cumulative_product = np.cumprod(arr9, axis=1)

"""print("\nArray")
print(arr7)
print("\nMean:")
print(arr7.mean())
print("\nStandard deviation:")
print(arr7.std())
print("\nVariance:")
print(arr7.var())"""


# BASIC ARRAY STATISTICAL METHODS (page 125)
people_data = {
    "Rose": {"age": 18, "sex": True, "income": 650, "children": 0, "pets": 0},
    "Victor": {"age": 31, "sex": False, "income": 1_150, "children": 0, "pets": 0},
    "Spyros": {"age": 58, "sex": False, "income": 1_200, "children": 1, "pets": 2},
    "Dimitra": {"age": 50, "sex": True, "income": 450, "children": 3, "pets": 0},
    "Billy": {"age": 56, "sex": False, "income": 3_220, "children": 2, "pets": 4},
    "Marie": {"age": 45, "sex": True, "income": 1_950, "children": 1, "pets": 1},
}

incomes = np.array(
    [
        inner_dict[category]
        for name, inner_dict in people_data.items()
        for category in inner_dict
        if category == "income"
    ]
)

age_values = np.array(
    [
        inner_dict[category]
        for name, inner_dict in people_data.items()
        for category in inner_dict
        if category == "age"
    ]
)

age_men = np.array(
    [
        inner_dict["age"]
        for name, inner_dict in people_data.items()
        for category in inner_dict
        if category == "sex"
        if inner_dict[category]
    ]
)

age_women = np.array(
    [
        inner_dict["age"]
        for name, inner_dict in people_data.items()
        for category in inner_dict
        if category == "sex"
        if not inner_dict[category]
    ]
)

income_men = np.array(
    [
        inner_dict["income"]
        for name, inner_dict in people_data.items()
        for category in inner_dict
        if category == "sex"
        if not inner_dict[category]
    ]
)

income_women = np.array(
    [
        inner_dict["income"]
        for name, inner_dict in people_data.items()
        for category in inner_dict
        if category == "sex"
        if inner_dict[category]
    ]
)

children = np.array(
    [
        inner_dict[category]
        for name, inner_dict in people_data.items()
        for category in inner_dict
        if category == "children"
    ]
)

sexes = np.array(
    [
        "Woman" if inner_dict[category] else "Man"
        for names, inner_dict in people_data.items()
        for category in inner_dict
        if category == "sex"
    ]
)


total_number_of_children_per_sex = {}
total_number_of_children_per_sex.setdefault("Men", 0)
total_number_of_children_per_sex.setdefault("Women", 0)

for name, inner_dict in people_data.items():
    for category in inner_dict:
        if category == "sex":
            if inner_dict[category]:
                total_number_of_children_per_sex["Women"] += inner_dict["children"]
            elif not inner_dict[category]:
                total_number_of_children_per_sex["Men"] += inner_dict["children"]

# print(total_number_of_children_per_sex)

age_men_2 = list()
age_women_2 = list()

# the advantage over the list comprehensions here is one single loop for both created lists
for name, inner_dict in people_data.items():
    for category in inner_dict:
        if category == "sex":
            if inner_dict[category]:
                age_women_2.append(inner_dict["age"])
            elif not inner_dict[category]:
                age_men_2.append(inner_dict["age"])

age_men_2 = np.array(age_men_2)
age_women_2 = np.array(age_women_2)

ages_and_incomes = np.array(list(zip(age_values, incomes))).reshape(len(age_values), 2)
# ages_and_incomes = people_data[:, [0, 2]]  # todo: this can replace the line above, if people_data was an np.array
children_per_sex = np.array(list(zip(children, sexes))).reshape(len(children), 2)

"""print("\nIncomes (sorted):")
print(sorted(incomes))
print("\nMean:")
print(round(incomes.mean(), 2))
print("\nStandard deviation:")
print(round(incomes.std()))"""


"""print("\nAverage")

print("\n\tAge: All")
print("\t\t" + str(round(np.float16(ages_and_incomes[:, [0]].mean()), 1)))

print("\n\tAge: Men")
print("\t\t" + str(round(np.float16(age_men.mean()), 1)))

print("\n\tAge: Women")
print("\t\t" + str(round(np.float16(age_women.mean()), 1)))

print("\n\tIncome: All")
print("\t\t" + str(round(np.float16(income_women.mean()), 1)))

print("\n\tIncome: Men")
print("\t\t" + str(round(np.float16(ages_and_incomes[:, [1]].mean()), 1)))

print("\n\tIncome: Women")
print("\t\t" + str(round(np.float16(income_women.mean()), 1)))

print("\nAges and incomes:")
print(ages_and_incomes)
print("\nchildren per Sex")
print(children_per_sex)"""

people_names = np.array([name for name in people_data])

people_data2 = np.array(
    [
        [18, True, 650, 0, 0],
        [31, False, 1_150, 0, 0],
        [58, False, 1_200, 1, 2],
        [50, True, 450, 3, 0],
        [56, False, 3_220, 2, 4],
        [45, True, 1_950, 1, 1],
    ],
    dtype=np.int16
)

names_and_data_as_tuples = np.array(list(zip(people_names, people_data2)), dtype=tuple)
# print(names_and_data_as_tuples)

children_list = people_data2[:, 3]
total_children = children_list.cumsum()[-1]  # the hell

ages_and_incomes2 = people_data2[:, [0, 2]]
ages_and_children = people_data2[:, [0, 3]]
incomes_and_pets = people_data2[:, [2, 4]]

# print("\nSubset query:")
# print(incomes_and_pets)
# print("\n")
# print(incomes_and_pets[:, 1])
# print(list(zip(sex_list, children_list)))


# METHODS FOR BOOLEAN ARRAYS (page 126)

my_array = np.array([0, 2, 3, 5, 6, -2, -5, -1, 0])

print("Sum of all values greater than 2:", my_array[(my_array > 2)].sum())

gt_zero_bools = my_array > 0
print("Values that are greater than 0: ", my_array[gt_zero_bools])

print("Any values are True?", gt_zero_bools.any())
print("All values are True?", gt_zero_bools.all())

# SORTING (page 127)
"""The .sort() method sorts arrays in-place.

The top-level np.sort() method returns a sorted copy of the array.


A quick-and-dirty way to compute the quantiles of an array is to
sort it and select the value at a particular rank:

large_arr = np.random.randn(1000)

large_arr.sort()

result = large_arr[int(0.05 * len(large_arr))]
"""

# UNIQUE AND OTHER SET LOGIC (page 127)
"""np.unique() returns the unique elements of the array. 

Equivalent of the Python sorted(set(my_array))

Array set operations (page 128)

unique(x)               Compute the sorted, unique elements in x
intersect1d(x, y)       Compute the sorted, common elements in x and y
union1d(x, y)           Compute the sorted union of elements
in1d(x, y)              Compute a boolean array indicating whether each element of x is contained in y
setdiff1d(x, y)         Set difference, elements in x that are not in y
setxor1d(x, y)          Set symmetric differences; elements that are in either of the arrays, but not both

"""


# FILE INPUT AND OUTPUT WITH ARRAYS (page 128)
"""
Saved on disk either in binary or text form.
In binary by default with the extension .npy

save_array = np.arange(1, 16)
np.save('Data/some_array', save_array)

loaded_array = np.load('Data/some_array.npy')

Save multiple arrays by passing them as keyword arguments using np.savez(), saving as .npz

np.savez('array_archive.npz', a=arr, b=arr)

When loading an .npz file, you get back a dict-like object that loads the individual
arrays lazily:

arch = np.load('array_archive.npz')
print(arch['b'])
>>> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

"""

# LINEAR ALGEBRA (page 129)
"""
page 130-131

.diag       Return the diagonal (or off-diagonal) elements of a square matrix as a 1D array, or convert a 1D array into a
square matrix with zeros on the off-diagonal

.dot        Matrix multiplication
.trace      Compute the sum of the diagonal elements
.det        Compute the matrix determinant
.eig        Compute the eigenvalues and eigenvectors of a square matrix
.inv        Compute the inverse of a square matrix
.pinv       Compute the Moore-Penrose pseudo-inverse of a matrix
.qr         Compute the QR decomposition
.svd        Compute the singular value decomposition (SVD)
.solve      Solve the linear system Ax = b for x, where A is a square matrix
.lstsq      Compute the least-squares solution to Ax = b

"""

# PSEUDORANDOM NUMBER GENERATION (page 131)
example = np.random.normal(size=(4, 4))  # Creates data using the standard normal distribution

# It is possible to change the RNG seed using np.random.seed()
# It is also possible to override default, global behaviour on the seed by using rng = np.random.RandomState(1234)
# then, rng.randn(x)

"""
Partial list of numpy.random functions (page 132)

seed            Seed the random number generator
permutation     Return a random permutation of a sequence, or return a permuted range
shuffle         Randomly permute a sequence in-place
rand            Draw samples from a uniform distribution
randint         Draw random integers from a given low-to-high range
randn           Draw samples from a normal distribution with mean 0 and standard deviation 1 (MATLAB-like interface)
binomial        Draw samples from a binomial distribution
normal          Draw samples from a normal (Gaussian) distribution
beta            Draw samples from a beta distribution
chisquare       Draw samples from a chi-square distribution
gamma           Draw samples from a gamma distribution
uniform         Draw samples from a uniform [0, 1) distribution
"""

sequence = np.arange(1, 7)
np.random.shuffle(sequence)

a = np.random.randn(10)

