
def main():

    # ZIP TO UNPACK
    actors = [("Brad", "Pitt"), ("Al", "Paccino"), ("Robert", "De Niro")]

    first_names, last_names = zip(*actors)
    list(zip(first_names, last_names))

    squares = [(2, 4), (3, 9), (4, 16), (5, 25)]
    a, b = zip(*squares)
    # print(a, b)


if __name__ == "__main__":
    main()
