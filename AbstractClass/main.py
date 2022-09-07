from abc import ABC, abstractmethod
from random import randint, triangular
from bisect import bisect, insort

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


# DATA ANALYSIS
def count_occurrence(my_list: list) -> dict:
    return {x: my_list.count(x) for x in my_list}


def count_number_of_duplicate_values(my_list: list) -> int:
    return len(my_list) - len(set(my_list))


def find_duplicate_values(my_list: list) -> tuple:
    return tuple(set([x for x in my_list if my_list.count(x) > 1]))


def find_unique_values(my_list: list) -> tuple:
    return tuple(x for x in my_list if my_list.count(x) == 1)


def main():

    # ABSTRACT CLASSES AND METHODS
    employee_1 = Dev("Victor", "Kaklamanis")
    # print(employee_1.display_name())

    # REFACTOR EXERCISE (COMPLETED)
    this_rectangle = Rectangle(length=3, width=7)
    # print(this_rectangle.area())
    # print(this_rectangle.perimeter())

    # DUPLICATES / DATA ANALYSIS EXERCISE (COMPLETED)
    my_list = [randint(1, 25) for _ in range(120)]
    my_list.extend([2, 2])

    duplicates = find_duplicate_values(my_list)
    frequency_dict = count_occurrence(my_list)

    # MAP() FUNCTION
    upper_bounds = [randint(1, 20) for _ in range(20)]
    lower_bounds = [randint(19, 40) for _ in range(20)]
    results = list(map(triangular_distribution, upper_bounds, lower_bounds))
    # print(results)

    # BISECT
    other_list = [1, 3, 1, 6, 4, 6, 2, 2]
    sorted_list = sorted(other_list)
    # print(sorted_list)
    bisect(sorted_list, 5)
    # bisect implements binary search and returns the index at which
    # the element has to be placed for the iterable to remain in order
    sorted_list_copy = sorted_list.copy()
    insort(sorted_list_copy, 5)
    # insort actually inserts the element at the corresponding index

    for element in set(sorted_list_copy):
        if element not in set(sorted_list):
            print(element)


if __name__ == "__main__":
    main()
