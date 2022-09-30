import pandas as pd
from pandas import Series, DataFrame

DECO = "-" * 50
NL = "\n"

my_dictionary_of_ds_weapons = {
    "name": ["Short Sword", "Mace", "Farron Ultra Greatsword"],
    "damage": [2, 5, 0],
    "class": ["Sword", "Mace", "Ultra Greatsword"],
    "is_unique": [False, False, True],
    "upgrade_level": [0, 1, 1]
}

data = DataFrame(my_dictionary_of_ds_weapons)

"""
DataFrame

    attributes
        columns
        
        
    methods
        head(x: int or None)
        tail(x: int or None)
        info()
        


access rows as if calling the index of a dict
    DataFrame['row_name']
        
access columns via      
"""

print(f"{NL}Data.info(){NL}")
print(data.info())
print(DECO + NL)
print(f"Data['class']{NL}")
print(data["class"])
print(DECO + NL)
print(f"{NL}Data.columns{NL}")
print(data.columns)
print(DECO + NL)
print(f"{NL}Data{NL}")
print(data)
