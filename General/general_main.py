
def main():

    # ZIP TO UNPACK
    actors = [("Brad", "Pitt"), ("Al", "Paccino"), ("Robert", "De Niro")]

    first_names, last_names = zip(*actors)
    list(zip(first_names, last_names))

    squares = [(2, 4), (3, 9), (4, 16), (5, 25)]
    a, b = zip(*squares)
    # print(a, b)

    # DICTS
    my_dict_1 = {"a": 1, "b": 2, "c": 3, "d": 7}
    my_dict_1.update({"d": 4, "e": 5})  # values to already existing keys are replaced by the new values
    del my_dict_1["e"]  # can be used instead of .pop() if the value will not be used
    # print(my_dict_1)


if __name__ == "__main__":
    main()
