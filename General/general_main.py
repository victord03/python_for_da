
def main():

    # ZIP TO UNPACK (page 74)
    actors = [("Brad", "Pitt"), ("Al", "Paccino"), ("Robert", "De Niro")]

    first_names, last_names = zip(*actors)
    list(zip(first_names, last_names))

    squares = [(2, 4), (3, 9), (4, 16), (5, 25)]
    a, b = zip(*squares)
    # print(a, b)

    # DICTS (page 74)
    my_dict_1 = {"a": 1, "b": 2, "c": 3, "d": 7}
    my_dict_1.update({"d": 4, "e": 5})  # values to already existing keys are replaced by the new values
    del my_dict_1["e"]  # can be used instead of .pop() if the value will not be used

    value = my_dict_1.get("f", "Not Found")  # .get() can take a default value as a second parameter, in case the key does not exist

    popped_value = my_dict_1.pop("f", "No such key exists")

    # .setdefault() (page 77)

    """Imagine wanting to create a dictionary with the first letter of words as keys 
    and a list with all the words that start with that letter as values.
    There are two ways to do it. A normal loop with checks and .setdefault()
    dict method"""

    words = ['apple', 'bat', 'bar', 'atom', 'book']

    by_letter = {}

    # using regular loop and checks
    for word in words:

        letter = word[0]

        if letter not in by_letter:
            by_letter[letter] = [word]

        else:
            by_letter[letter].append(word)

    by_letter.clear()

    # using .setdefault()
    for word in words:

        letter = word[0]
        by_letter.setdefault(letter, []).append(word)

    # print(by_letter)

    # also see from collections import defaultdict (page 77)

    # HASHABILITY (page 77)

    # it is possible to check if an object is mutable if it raises no error from the hash function

    # print(hash("string"))
    # print(hash((1, 2, (2, 3))))
    # hash((1, 2, [3, 4]))  # raises an error because a list is mutable

    # SETS (page 78)
    
    """Sets are like dicts but only keys and no values.
    Support mathematical operations such as union, intersection, 
    difference, and symmetric difference.
    See page 79 for a table with python set operations."""

    set_1 = {1, 2, 3, 4, 5}
    set_2 = {5, 22, 23, 24, 24}
    list_1 = [0, 9, 10, 5]

    # .union() returns a set containing the distinct elements occurring in either container
    # .union() can take multiple arguments
    set_union = set_1.union(set_2)  # can be written as set_1 | set_2

    # .intersection() returns a set containing the elements occurring in all containers
    set_inter = set_1.intersection(set_2)  # can be written as set_1 & set_2

    set_4 = {3, 4}
    set_4.issubset(set_1)
    set_union.issuperset(set_2)

    set_5 = {3, 2, 5, 4, 1}
    statement = set_1 == set_5  # strict equality (all elements must be present in both)

    # COMPREHENSIONS
    strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
    set(map(len, strings))  # returns a set (only unique values) with the length of strings as elements


if __name__ == "__main__":
    main()
