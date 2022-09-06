from enum import Enum


class Choices(Enum):
    OPTION_1 = 1
    OPTION_2 = 2
    OPTION_3 = 3
    OPTION_4 = 4


def main():

    a = Choices(2)
    print(type(a))
    print(a, a.value)


if __name__ == "__main__":
    main()
