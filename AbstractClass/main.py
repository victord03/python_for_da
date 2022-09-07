import random
from abc import ABC, abstractmethod

from random import randint,triangular


class Employee(ABC):
    name: str
    last_name: str
    email: str

    def __init__(self, name: str, last_name: str):
        self.name = name
        self.last_name = last_name

    @abstractmethod
    def create_email(self):
        ...


class Dev(Employee):

    def create_email(self):
        self.email = self.name + "." + self.last_name + "@company.com"

    def display_name(self) -> str:

        if not self.__dict__.get('email'):
            self.create_email()

        name = f"\nNAME: {self.name}"
        last_name = f"\nLAST NAME: {self.last_name}"
        email = f"\nEMAIL: {self.email}"
        return name + last_name + email


class Shape(ABC):

    @abstractmethod
    def area(self):
        raise NotImplementedError

    @abstractmethod
    def perimeter(self):
        raise NotImplementedError


class Rectangle(Shape):

    length: int
    width: int

    def __init__(self, length: int, width: int) -> None:
        self.length = length
        self.width = width

    def area(self) -> int:
        return self.length * self.width

    def perimeter(self) -> int:
        return 2 * (self.length + self.width)


def triangular_distribution(lower_bound: float, upper_bound: float) -> float:
    return round(triangular(lower_bound, upper_bound), 2)


def main():

    # ABSTRACT CLASSES AND METHODS
    employee_1 = Dev("Victor", "Kaklamanis")
    print(employee_1.display_name())

    # REFACTOR EXERCISE (COMPLETED)
    # this_rectangle = Rectangle(length=3, width=7)
    # print(this_rectangle.area())
    # print(this_rectangle.perimeter())

    # DUPLICATES EXERCISE (COMPLETED)
    # my_list = [randint(1, 25) for _ in range(4000)]
    # my_list.extend([2, 2])

    # print(my_list)
    # print()

    # duplicates = list(set(x for x in my_list if my_list.count(x) > 1))

    # print(duplicates)
    # print()

    # frequency_dict = {x: my_list.count(x) for x in sorted(my_list)}
    # print(frequency_dict)

    # MAP
    # upper_bounds = [randint(1, 20) for _ in range(20)]
    # lower_bounds = [randint(19, 40) for _ in range(20)]
    # results = list(map(triangular_distribution, upper_bounds, lower_bounds))
    # print(results)


if __name__ == "__main__":
    main()
