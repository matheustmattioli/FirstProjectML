# Nomes e RAs:
# Gabriel Penajo Machado,    769712
# Matheus Ramos de Carvalho, 769703
# Matheus Teixeira Matioli,  769783

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


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

    # dataframe
    df = pd.read_csv(
        'imports-85.data', header=None, names=column_headers)
        

    # pré-processamento

    # Atribuindo o valor NaN para informações ausentes
    imp_const = SimpleImputer(missing_values='?', strategy='constant', fill_value=np.nan)
    df = pd.DataFrame(imp_const.fit_transform(df), columns=column_headers)

    # Convertendo atributos categóricos para atributos numéricos
    enc = preprocessing.OrdinalEncoder()
    categorical_columns = [
        "make", "fuel-type", "aspiration",  "num-of-doors","body-style",
        "drive-wheels", "engine-location", "engine-type", "num-of-cylinders",
        "fuel-system"
    ]
    
    df[categorical_columns] = enc.fit_transform(df[categorical_columns])

    # Dividindo os casos de teste e casos de treino
    # X = assim
    # y = assado
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.33, random_state=42)


    # Existem alguns dados faltando, portanto, vamos tratar isso
    # imp_mean = SimpleImputer(missing_values='?', strategy='most_frequent')
    # imp_mean.fit(df)

    print(df)
