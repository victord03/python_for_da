# import matplotlib.pyplot as plt
from numpy.random import randn
from datetime import datetime as dt

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


# DATA ANALYSIS
this_list = [0, 1, 2, 3, 2, 4, 2, 6, 0, 2, 6]


def count_occurrence(my_list: list) -> dict:
    return {x: my_list.count(x) for x in my_list}


print(count_occurrence(this_list))


def count_number_of_duplicate_values(my_list: list) -> int:
    return len(my_list) - len(set(my_list))


print(count_number_of_duplicate_values(this_list))


def find_duplicate_values(my_list: list) -> tuple:
    return tuple(set([x for x in my_list if my_list.count(x) > 1]))


print(find_duplicate_values(this_list))


def find_unique_values(my_list: list) -> tuple:
    return tuple(x for x in my_list if my_list.count(x) == 1)


print(find_unique_values(this_list))

