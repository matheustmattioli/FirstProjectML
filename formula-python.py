# Nomes e RAs:
# Gabriel Penajo Machado,    769712
# Matheus Ramos de Carvalho, 769703
# Matheus Teixeira Matioli,  769783

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


# dataset src: https://archive.ics.uci.edu/ml/support/Automobile#e746c17201da2dd72583f7b9b0c2a6ba310412f4

if __name__ == '__main__':
    # Opening dataset
    column_headers = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
        "num-of-doors", "body-style", "drive-wheels",
        "engine-location", "wheel-base", "length", "width", "height",
        "curb-weight", "engine-type", "num-of-cylinders", "engine-size",
        "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
        "peak-rpm", "city-mpg", "highway-mpg", "price"
    ]

    DataFrame = pd.read_csv(
        'imports-85.data', header=None, names=column_headers)
    print(DataFrame)
