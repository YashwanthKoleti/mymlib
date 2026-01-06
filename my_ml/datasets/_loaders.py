import numpy as np
import pandas as pd
import os
import requests
# from keras.datasets import fashion_mnist

def load_spambase(data_folder = "data",
                  filename = "spambase.csv",
                  download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"):
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, data_folder)
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        response = requests.get(download_url)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)


    data = np.loadtxt(file_path, delimiter=",")

    X = data[:, :-1]
    y = data[:, -1]

    return X,y

# def load_fashion(data_folder = "", 
#                  filename = "",
#                  download_url = "",
#                  kind = 'Train',
#                  normalize = True):
    
#     if kind not in ('Train', 'Test'):
#         raise ValueError("kind must be 'Train' or 'Test'")
    
#     file_path = os.path.join(data_folder, filename)
#     (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#     if normalize:
#         x_train = x_train.astype('float32') / 255.0
#         x_test = x_test.astype('float32') / 255.0

#     if kind == 'Train':
#         return (x_train, y_train)

#     return (x_test, y_test)