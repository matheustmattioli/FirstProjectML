# Nomes e RAs:
# Gabriel Penajo Machado,    769712
# Matheus Ramos de Carvalho, 769703
# Matheus Teixeira Matioli,  769783

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, max_error, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# import matplotlib.pyplot as plt

# dataset src: https://archive.ics.uci.edu/ml/support/Automobile#e746c17201da2dd72583f7b9b0c2a6ba310412f4
DEBUG = 0

scaler = preprocessing.StandardScaler()

# Recebe um DataFrame e retorna um Dataframe normalizado
def normalize_data(dataframe):
    dataframe = scaler.fit_transform(dataframe)

    return dataframe

# Printa uma mensagem se a flag DEBUG estiver habilitada
def debug(message):
    if (DEBUG == 1):
        print(message)

# Atribui valores aos NaN
def imputate_data(X_train, X_test, y_train, y_test, col_X, col_y):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')

    X_train = pd.DataFrame(imp_mean.fit_transform(
        X_train), columns=col_X)
    y_train = pd.DataFrame(imp_mean.fit_transform(
        y_train), columns=col_y)

    X_test = pd.DataFrame(imp_mean.fit_transform(
        X_test), columns=col_X)
    y_test = pd.DataFrame(imp_mean.fit_transform(
        y_test), columns=col_y)

    return (X_train, X_test, y_train, y_test)

# Ordena os pares de preços preditos e de preços reais
def sort_both(y_best_pred_ordered, y_test_ordered):
    # Ordenando para melhorar a visualização
    zipped_pairs = zip(y_best_pred_ordered, y_test_ordered)
    tuple_list = list(zip(*sorted(zipped_pairs)))
    return tuple_list[0], tuple_list[1]
    
# Realiza o pré-processamento do DataFrame df
def preprocess_dataset(df, drop = False):
    # Atribuindo o valor NaN para informações ausentes
    imp_const = SimpleImputer(
        missing_values='?', strategy='constant', fill_value=np.nan)
    df = pd.DataFrame(imp_const.fit_transform(df), columns=column_headers)
    
    if drop:
        df.dropna(inplace=True)

    # Normalizando os dados contínuos (ignora valores NaN)
    to_normalize = []

    for attr in column_headers:
        if attr not in categorical_columns and attr != 'price':
            to_normalize.append(attr)

    df[to_normalize] = normalize_data(df[to_normalize])

    # Convertendo atributos categóricos para atributos numéricos
    enc = preprocessing.OrdinalEncoder()
    df[categorical_columns] = enc.fit_transform(df[categorical_columns])

    # Dividindo os casos de teste e casos de treino
    X = df[column_headers_X]
    y = df[['price']]

    # Valores contínuos também incluem a coluna y. Por isso, vamos separar a
    # lista to_normalize em 2 slices
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # Existem alguns dados faltando, portanto, vamos tratar isso
    
    # Atribuir valores aos NaN
    (X_train, X_test, y_train, y_test) = imputate_data(X_train, X_test,
        y_train, y_test, column_headers_X, column_headers_y)

    # Converte os atributos categóricos para int
    X_train[categorical_columns] = X_train[categorical_columns].astype(int)
    X_test[categorical_columns] = X_test[categorical_columns].astype(int)
    
    return (X_train, X_test, y_train, y_test)


def plot_compare_graph(y_test, y_best_pred_ordered, y_other_pred = []):
    # Ordenando para melhorar a visualização
    y_best_pred_ordered, y_test_ordered = sort_both(
        y_best_pred_ordered, y_test)
    
    if len(y_other_pred) != 0:
        y_other_pred, y_test_ordered = sort_both(
        y_other_pred, y_test)
        plt.plot(range(len(y_other_pred)),
            y_other_pred, label='Preco predito Random Forest', color='blue')
        plt.plot(range(len(y_best_pred_ordered)),
            y_best_pred_ordered, label='Preco predito KNN', color='red')
    else:
        plt.plot(range(len(y_best_pred_ordered)),
            y_best_pred_ordered, label='Preco predito', color='red')
    plt.scatter(range(len(y_test_ordered)), y_test_ordered,
        label='Preco real', color='#00FF00')

    plt.legend()
    plt.show()

# Regressão utilizando K-Nearest Neighbors
def KNeighbors(X_train, X_test, y_train, y_test, show=True):
    # Vamos aplicar o GridSearch para buscar os hiperparametros ótimos para o
    # algoritmo de regressão
    param_grid = {
        'weights': ['uniform', 'distance'],
        'n_neighbors': range(1, 15, 2),
        'metric': ['euclidean', 'manhattan']
    }
    knr = KNeighborsRegressor()
    gs = GridSearchCV(knr, param_grid=param_grid)
    gs.fit(X_train, y_train)
    # Análise qualitativa do desempenho do melhor classificador
    print('KNN:')
    print(f'Melhor score obtido: {gs.best_score_}')
    # Atributos do melhor classificador
    print(f'Parametros do melhor classificador:\n{gs.best_estimator_}')
    print(f'Parametros detalhados do melhor classificador:\n{gs.best_params_}')

    # Teste do classificador utilizando os melhores parâmetros
    knr2 = KNeighborsRegressor(
        metric=gs.best_params_['metric'],
        n_neighbors=gs.best_params_['n_neighbors'],
        weights=gs.best_params_['weights']
    )
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
    print(f"{scores.mean():.2f} acuracia no K-NN com desvio padrao de {scores.std():.2f}")

    if not show:
        return y_best_pred

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
    plt.plot(range(1, 20, 2), errors_r2, label='R2')
    plt.plot(range(1, 20, 2), errors_mae/min(errors_mae),
        label='Erro absoluto medio')
    plt.plot(range(1, 20, 2), errors_mse/min(errors_mse),
        label='Erro quadratico medio')
    plt.legend()
    plt.xticks(np.arange(1, 21, 2))
    plt.show()
    # Plot dos scores e dos desvios padrões para o KNN
    # com k variando entre [1, 20]
    plt.errorbar(x=range(1, 20, 2), y=score_neigh, yerr=std_neigh)
    plt.xticks(np.arange(1, 21, 2))
    plt.show()

    # Plot do melhor resultado
    plot_compare_graph(y_test['price'], y_best_pred)
    return y_best_pred

# Regressão utilizando Random Forest
def RandomForest(X_train, X_test, y_train, y_test, show=True):
    # Vamos aplicar o GridSearch para buscar os hiperparametros ótimos para o
    # algoritmo de regressão.
    param_grid = {
        'criterion': ['squared_error', 'absolute_error', 'poisson'],
        'bootstrap': [True, False],
        'max_depth': [10, 70, 80, 90, 100, 110],         
    }
                
    forest = RandomForestRegressor(random_state=42)
    gs = GridSearchCV(forest, param_grid=param_grid)
    gs.fit(X_train[column_headers_X], y_train['price'])

    print('Random Forest:')
    # Análise qualitativa do desempenho do melhor classificador
    print(f'Melhor score obtido: {gs.best_score_}')
    # Atributos do melhor classificador
    print(f'Parametros do melhor classificador:\n{gs.best_estimator_}')
    print(f'Parametros detalhados do melhor classificador:\n{gs.best_params_}')
    
    forest = RandomForestRegressor(
        bootstrap=gs.best_params_['bootstrap'],
        criterion=gs.best_params_['criterion'],
        max_depth=gs.best_params_['max_depth'],
        random_state=42
    )
    forest.fit(X_train, y_train['price'])
    y_pred = forest.predict(X_test)

    scores = cross_val_score(forest, X_train, y_train['price'], cv=5)
    print(scores)
    print(f"{scores.mean():.2f} acuracia no Random Forest com desvio padrao de {scores.std():.2f}")

    if show:
        plot_compare_graph(y_test['price'], y_pred)
    return y_pred


if __name__ == '__main__':
    # Opening dataset
    column_headers = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
        "num-of-doors", "body-style", "drive-wheels", "engine-location",
        "wheel-base", "length", "width", "height", "curb-weight",
        "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
        "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
        "city-mpg", "highway-mpg", "price"
    ]
    categorical_columns = [
        "symboling", "make", "fuel-type", "aspiration",  "num-of-doors",
        "body-style", "drive-wheels", "engine-location", "engine-type",
        "num-of-cylinders", "fuel-system"
    ]
    column_headers_X = column_headers[0:-1]
    column_headers_y = column_headers[-1:]
    
    # DataFrame
    df = pd.read_csv(
        'imports-85.data', header=None, names=column_headers)

    # Pré-processamento
    valid_options = ['Y','N']
    drop = input('Ignorar valores nulos? (Y/N)\n')
    while drop not in valid_options:
        drop = input('Digite uma opcao valida: (Y/N)\n')
    drop = True if drop == 'Y' else False
    (X_train, X_test, y_train, y_test) = preprocess_dataset(df, drop)

    valid_options = range(0,3)
    show = int(input('Realizar qual algoritmo?\n0-KNN\n1-Random Forest\n2-Ambos\n'))
    while show not in valid_options:
        show = int(input('Digite um numero valido:\n0-KNN\n1-Random Forest\n2-Ambos\n'))
    # KNN
    if show == 0:
        kn_pred = KNeighbors(X_train, X_test, y_train, y_test)
    # Random Forest
    elif show == 1:
        forest_pred = RandomForest(X_train, X_test, y_train, y_test)
    # KNN e Random Forest
    else:
        forest_pred = RandomForest(X_train, X_test, y_train, y_test, False)
        print(30*"-")
        knn_pred = KNeighbors(X_train, X_test, y_train, y_test, False)
        plot_compare_graph(y_test['price'], knn_pred, forest_pred)


# MMMMMMMMMMMMMMMWWWWWWWWMWWWMWWWWMMMWNXK000KKXNWWMMWWMMWWWWWWWWWWWWWWWWWWWWWWWWWW
# MMMMMMMMMMMMMMMWWWWWWWWMWWWMMWNX0kolc::;;;;;;:cloxOXNWWWWWWWWWWWWWWWWWWWWWWWWWWW
# MMMMMMMMMMMMMMMWWWWWWWWWWWWN0dc,...................,cxKNWWWWWWWWWWWWWWWWWWWWWWWW
# MMMMMMMMMMMMMMMWWWWWWWWWN0d:'.........................,lONWWWWMWWWWWWWWWWWWWWWWW
# WWWWWWWMMMMMWWWMWWWWWN0d:'.     ...   ..................'ckXWWWWWWWWWWWWWWWWWWWW
# WWWWWWWWMMMWWWWWWWWWKd,..           ..........       ......:ONWWWWWWWWWWWWWWWWWW
# WWWWWWWWWWWWWWWWWWWKd;..          ..         ..    ...   ...'oXWWWWWWWWWWWWWWWWW
# WWWWWWWWWWWWWWWWWXkc'...........................       ..  ...oNWWWWWWWWWWWWWWWW
# WWWWWWWWWWWWWWWWNx,.......',,''',,,,,',''''........      .  ..,kWWWWWWWWWWWWWWWW
# WWWWWWWWWWWWWWWW0c'....',,;:cc:::ccc:::;;;;,,''......       ...cKWWWWWWWWWWWWWWW
# WWWWWWWWWWWWWWWXd'.. .':::ccclllllollcc:::::;;;;;,,,'...........dNWWWWWWWWWWWWWW
# WWWWWWWWWWWWWWWO:.   .;ccllloddxdxxxxxxddoolccc:cccc:;;;,,'..  .:KWWWWWWWWWWWWWW
# WWWWWWWWWWWWWWNd'.   ':looddxxxkkOOOOOOOOkkxxddooloooolcccc:'.  'kWWWWWWWWWWWWWW
# WWWWWWWWWWWWWWXl..  .'codddxxxxxkkkkkkkkxxxxxxxxxddddddddoll:.  .cKWWWWWWWWWWWWW
# WWWWWWWWWWWWWWXc..  .,ldddddddolooodddddddddddoolllcccllllodl'  .:KWWWWWWWWWWWWW
# WWWWWWWWWWWWWWNd..  .:odo:,,''.....',:cloodooc;,.........';lo;. .,0WWWWWWWWWWWWW
# WWWWWWWWWWWWWWW0;.. .ldc,....      ...':lodoc,...      ...';lc. .'xNWNWWWWWWWWWW
# WWWWWWWWWWWWNWW0:.. ;ol;''...     ....,ldxkxo:....    ...',;ll' ..lXWWWWNWWWWWWW
# WWWWWWWWWWWWNNWKc. .cdc;'. ..    .....:dxkkkxl,....  .....';co;..'dNWNWWNWWWWWWW
# WWWWWWWWWWWNNNWXo'.'odc;'...........,:oxxkOkxoc,'....''''',;coc'.:0WNNWWWWWWWWWW
# WWWWWWWWWWWNNNW0c,';odolccc::::::::clodxkkOkxdlcc:;;;;::cclloooc,cONNNWWWWWWWWWW
# WWWWWWWWWWWNWWWXo::cddddddddddddxddolodxxkkxddlloodddooooddooddl:ckXNNNNNWWWWWWW
# WWWWWWWWWWWNWWWXxlccoddddddxxxxxxdollloddxkxdollloddxdddddddddolclkXNNNNNNWWWWWW
# WWWWWWWWWWWNWWWNOollodddoddddxxxdolllooodxkxdolclloddddddddddolc:oOXXNNNNNNWWWWW
# WWWWWWWWWWWWWNWWKdooooooooooddodolloooooodxxdoooollooooddooodolccd0XXXXNNNNNWWWW
# WWWWWWWWWWWWWNNWXOdoooooooooooooollc:;;::cccc:::ccloooooooooooolokKKKXXXNNNNWWWW
# WWWWWWWWWWWWWNNNNXkdddooooloooooddl:'....',,,,',;clooooooooooolox0KKKXXXNNNNNWWW
# WWWWWWWWWWWWWNNNNN0xoooooolloooooooc:;,'.'''',,;::cloooooooooddxO00KXXXXXNNNNNNN
# WWWWWWWWWWWWWNNNNNN0kxxdollllllcccccccc:;;,,,;;::::ccllllllodxkO000KKXXXXXNNNNNN
# WWWWWWWWWWWWWNNNNNXXXKOxolllc:;,,,,;;;;;;;;,,,,,,,,',;cclllodxkOO00KKKXXXXXNNNNN
# WWWWWWWWWWWWWNNWNNNXXX0kolll:,'...'',;;,,'''''''''''',;:cclodkkOO00KKKXXXXXXNNNN
# WWWWWWWWWWWWWNNWNNNXXXKOdlcl:;::::::ccccc::;;;;;;:::;;;:ccloxkkOO00KKKKXXXXXXNNN
# WWWWWWWWWWWWWNNNNNXXXXKKklccc:cllcc:;;,,,,,,,,;:cc::;;:::cldxxkOO000KKKKXXXXXNNN
# WWWWWWWWWWWWWNNNNNXXXXKK0xl:cccccccc:;,,,,,,,;::c:::;;:::codxxkOOO00KKKKXXXXXXNN
# WWWWWWWNNNWWNNNNNNXXXKK000d::ccccloollllllccccllcc:::::::cldxkkOOO000KKKKXXXXXXX
# WWWWWNNNNNNNNNNNNXXXXKK000kl::cccllooddoooooooolcc::;;::ccldxkkkOO0000KKKXXXXXXX
# WWWNNNNNNNNNNNNNNNXXXKKK00Oo:;;:::ccllollllllcc:::;;;;;::ccdxkkkOO0000KKKKKXXXXX
# NNNNNNNNNNNNNNNNNNXXXKKK00Ooc:,,,,,,,;;::::;;,,,''',,;;:::codxxkOOOO000KKKKKXXXX
# NNNNNNNNNNNNNNNNNNXXXKK000koc:;,'................'',,;;:::clodxkkOOO0000KKKKKXXX
# NNNNNNNNNNNNNNNNNNXXXKKKOkdlcc:;,'..............'',,;;;:::cclll:cdk0O0000KKKKKXX
# NNNNNNNNNNNNNNNNNXXXK0xlcllccc::;,''.........''',,;;;;::::ccccc,..'cdO0000KKKKXK
# NNNNNNNNNNNNNNNNKOdc;'..,lllccc::;;,''....'''',,,;;;;::::cccccc:.   .':ldk0KKKKK
# NNNNNNNNNNXK0kdc,.      .clclcc::::;;,'''''',,,,;;;;;::::c::ccc;.       ..,cdk0K
# NNXXXXKOxl;'..           .cllccc:::::;,,,,,,,,,,;;;;;::::cc:c:;.             .;c
# 0kkxl;'.                  .:lolccc::::;;,,,,,,,,,;;;;::cccclc,.                 
# ...                        .,cllllc:c::;;;;,,,,,,;;:::cccllc,.                  
#                              ..;cllllcc::::;;,,,,;:::clllc,.                    
#                                 ..',;:::cc::::;;;::ccc:,..                      
#                                       .......'''''....                          