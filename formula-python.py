# Nomes e RAs:
# Gabriel Penajo Machado,    769712
# Matheus Ramos de Carvalho, 769703
# Matheus Teixeira Matioli,  769783

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import max_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

# import matplotlib.pyplot as plt

# dataset src: https://archive.ics.uci.edu/ml/support/Automobile#e746c17201da2dd72583f7b9b0c2a6ba310412f4
DEBUG = 0

scaler = preprocessing.StandardScaler()

# Recebe um DataFrame e retorna um Dataframe normalizado


def normalize_data(dataframe):
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
    X_test[categorical_columns] = X_test[categorical_columns].astype(int)

    debug(f"{X_train}\n=====\n{y_train}")

    # Vamos aplicar o GridSearch para buscar os hiperparametros ótimos para o
    # algoritmo de regressão
    param_grid = {'weights': ['uniform', 'distance'],
                  'n_neighbors': range(1, 15, 2),
                  'metric': ['euclidean', 'manhattan']}
    knr = KNeighborsRegressor()
    gs = GridSearchCV(knr, param_grid=param_grid)
    gs.fit(X_train, y_train)
    # Análise qualitativa do desempenho do melhor classificador
    print(f'Melhor score obtido: {gs.best_score_}')
    # Atributos do melhor classificador
    print(f'Parametros do melhor classificador:\n{gs.best_estimator_}')
    print(f'Parametros detalhados do melhor classificador:\n{gs.best_params_}')

    # Teste do classificador utilizando os melhores parâmetros
    knr2 = KNeighborsRegressor(
        metric='manhattan', n_neighbors=5, weights='distance')
    knr2.fit(X_train, y_train)
    y_best_pred = knr2.predict(X_test)
    y_pred = y_best_pred
    # Scores variados para avaliar o desempenho
    print(f'R2-score: {r2_score(y_test, y_pred)}')
    print(f'MSE: {mean_squared_error(y_test, y_pred)}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')

    # 5-Fold Cross Validation
    scores = cross_val_score(knr2, X_train, y_train, cv=5)
    print(scores)
    print(f"{scores.mean():.2f} acuracia com desvio padrao de {scores.std():.2f}")

    # Análise do KNN para k no intervalo [1, 20]
    errors_mae = []
    errors_mse = []
    errors_r2 = []
    score_neigh = []
    std_neigh = []

    for i in range(1, 20, 2):
        neigh = KNeighborsRegressor(
            metric='manhattan', n_neighbors=i, weights='distance')
        neigh.fit(X_train, y_train)

        y_pred = neigh.predict(X_test)
        errors_mae.append(mean_absolute_error(y_test, y_pred))
        errors_mse.append(mean_squared_error(y_test, y_pred))
        errors_r2.append(r2_score(y_test, y_pred))
        scores = cross_val_score(neigh, X_train, y_train, cv=5)
        score_neigh.append(scores.mean())
        std_neigh.append(scores.std())
    # Plot dos valores de cada métrica para melhor análise do resultado
    plt.plot(range(1, 20, 2), errors_mae, label='Erro absoluto medio')
    plt.plot(range(1, 20, 2), errors_mse, label='Erro quadratico medio')
    plt.plot(range(1, 20, 2), errors_r2, label="R2")
    plt.legend()
    plt.xticks(np.arange(1, 21, 2))
    plt.show()
    # Plot dos scores e dos desvios padrões para o knn com k variando entre [1, 20]
    plt.errorbar(x=range(1, 20, 2), y=score_neigh, yerr=std_neigh)
    plt.xticks(np.arange(1, 21, 2))
    plt.show()

    # Plot do melhor resultado

    y_test_ordered = y_test['price']
    y_best_pred_ordered = y_best_pred

    # Ordenando para melhorar a visualização
    zipped_pairs = zip(y_best_pred_ordered, y_test_ordered)
    tuple_list = list(zip(*sorted(zipped_pairs)))
    y_best_pred_ordered, y_test_ordered = tuple_list[0], tuple_list[1]

    plt.scatter(range(len(y_test_ordered)), y_test_ordered,
             label='Preco real', color='#00FF00')
    plt.plot(range(len(y_best_pred_ordered)),
                y_best_pred_ordered, label='Preco predito', color='red')

    plt.legend()
    plt.show()
