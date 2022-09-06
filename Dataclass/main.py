from dataclasses import dataclass


@dataclass
class Weapon:
    name: str
    damage: dict


def main():

    bkh = Weapon("BKH", {'physical': 10, 'magical': 0, 'fire': 0, 'lightning': 9})
    print(bkh)


if __name__ == "__main__":
    main()
