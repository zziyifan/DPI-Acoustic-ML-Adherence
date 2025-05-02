import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(transform=True):
    # Load y_train
    y_train = pd.read_csv(r'..\Features\y_train.csv', index_col=0)
    # Load y_test
    y_test = pd.read_csv(r'..\Features\y_test.csv', index_col=0)

    # Feature set 6: Combination 2
    X_train_6 = pd.read_csv(r'..\Features\X_train_6.csv', index_col=0)
    X_test_6 = pd.read_csv(r'..\Features\X_test_6.csv', index_col=0)
    

    if transform == True:
        minmax = MinMaxScaler()
        X_train_6 = minmax.fit_transform(X_train_6)
        X_test_6 = minmax.fit_transform(X_test_6)
        

    return (y_train, y_test, 
            X_train_6, X_test_6)

