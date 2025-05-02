import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(transform=True):
    # Load y_train
    y_train = pd.read_csv('features/y_train.csv', index_col=0)
    # Load y_test
    y_test = pd.read_csv('features/y_test.csv', index_col=0)

    # Feature set 1: Spectrogram
    X_train_1 = pd.read_csv('features/X_train_1.csv', index_col=0)
    X_test_1 = pd.read_csv('features/X_test_1.csv', index_col=0)

    # Feature set 2: Melspectrogram
    X_train_2 = pd.read_csv('features/X_train_2.csv', index_col=0)
    X_test_2 = pd.read_csv('features/X_test_2.csv', index_col=0)

    # Feature set 3: MFCC
    X_train_3 = pd.read_csv('features/X_train_3.csv', index_col=0)
    X_test_3 = pd.read_csv('features/X_test_3.csv', index_col=0)
    
    # Feature set 4: Chromagram
    X_train_4 = pd.read_csv('features/X_train_4.csv', index_col=0)
    X_test_4 = pd.read_csv('features/X_test_4.csv', index_col=0)

    # Feature set 5: Combination 1
    X_train_5 = pd.read_csv('features/X_train_5.csv', index_col=0)
    X_test_5 = pd.read_csv('features/X_test_5.csv', index_col=0)

    # Feature set 6: Combination 2
    X_train_6 = pd.read_csv('features/X_train_6.csv', index_col=0)
    X_test_6 = pd.read_csv('features/X_test_6.csv', index_col=0)
    

    if transform == True:
        minmax = MinMaxScaler()
        X_train_1 = minmax.fit_transform(X_train_1)
        X_test_1 = minmax.fit_transform(X_test_1)
        X_train_2 = minmax.fit_transform(X_train_2)
        X_test_2 = minmax.fit_transform(X_test_2)
        X_train_3 = minmax.fit_transform(X_train_3)
        X_test_3 = minmax.fit_transform(X_test_3)
        X_train_4 = minmax.fit_transform(X_train_4)
        X_test_4 = minmax.fit_transform(X_test_4)
        X_train_5 = minmax.fit_transform(X_train_5)
        X_test_5 = minmax.fit_transform(X_test_5)
        X_train_6 = minmax.fit_transform(X_train_6)
        X_test_6 = minmax.fit_transform(X_test_6)
        

    return (y_train, y_test, 
            X_train_1, X_test_1,
            X_train_2, X_test_2,
            X_train_3, X_test_3,
            X_train_4, X_test_4,
            X_train_5, X_test_5,
            X_train_6, X_test_6)

