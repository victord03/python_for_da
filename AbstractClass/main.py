from abc import ABC, abstractmethod as absmtd


class Employee(ABC):
    name: str
    last_name: str
    email: str

    def __init__(self, name: str, last_name: str):
        self.name = name
        self.last_name = last_name

    @absmtd
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


def main():
    employee_1 = Dev("Victor", "Kaklamanis")
    print(employee_1.display_name())


if __name__ == "__main__":
    main()
