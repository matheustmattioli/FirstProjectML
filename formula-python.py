# Nomes e RAs:
# Gabriel Penajo Machado,    769712
# Matheus Ramos de Carvalho, 769703
# Matheus Teixeira Matioli,  769783

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

# import matplotlib.pyplot as plt

# dataset src: https://archive.ics.uci.edu/ml/support/Automobile#e746c17201da2dd72583f7b9b0c2a6ba310412f4
DEBUG = 1


# Recebe um DataFrame e retorna um Dataframe normalizado
def normalize_data(dataframe):
    scaler = preprocessing.StandardScaler()
    dataframe = scaler.fit_transform(dataframe)
    
    return dataframe


def debug(message):
    if (DEBUG == 1):
        print(message)


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
    categorical_columns = [
        "symboling", "make", "fuel-type", "aspiration",  "num-of-doors", "body-style",
        "drive-wheels", "engine-location", "engine-type", "num-of-cylinders",
        "fuel-system"
    ]
    # DataFrame
    df = pd.read_csv(
        'imports-85.data', header=None, names=column_headers)

    # Pré-processamento

    # Atribuindo o valor NaN para informações ausentes
    imp_const = SimpleImputer(
        missing_values='?', strategy='constant', fill_value=np.nan)
    df = pd.DataFrame(imp_const.fit_transform(df), columns=column_headers)

    # Normalizando os dados contínuos (ignora valores NaN)
    to_normalize = []

    for attr in column_headers:
        if attr not in categorical_columns:
            to_normalize.append(attr)

    df[to_normalize] = normalize_data(df[to_normalize])

    # Convertendo atributos categóricos para atributos numéricos
    enc = preprocessing.OrdinalEncoder()
    df[categorical_columns] = enc.fit_transform(df[categorical_columns])

    # Dividindo os casos de teste e casos de treino
    X = df[column_headers[0:-1]]
    y = df[['price']]

    # Valores contínuos também incluem a coluna y. Por isso, vamos separar a
    # lista to_normalize em 2 slices
    column_headers_X = column_headers[0:-1]
    column_headers_y = column_headers[-1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    
    # Existem alguns dados faltando, portanto, vamos tratar isso
    # Atribuir valores aos NaN
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')

    X_train = pd.DataFrame(imp_mean.fit_transform(
        X_train), columns=column_headers_X)
    y_train = pd.DataFrame(imp_mean.fit_transform(
            y_train), columns=column_headers_y)

    X_test = pd.DataFrame(imp_mean.fit_transform(
        X_test), columns=column_headers_X)
    y_test = pd.DataFrame(imp_mean.fit_transform(
            y_test), columns=column_headers_y)

    # Converte os atributos categóricos para int
    X_train[categorical_columns] = X_train[categorical_columns].astype(int)
    y_test[categorical_columns] = y_test[categorical_columns].astype(int)
    

    debug(f"{X_train}\n=====\n{y_train}")
