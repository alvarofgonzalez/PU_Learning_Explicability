import pandas as pd
import pickle
import os
import numpy as np

def get_pickle(path, name):
    with open(path+name, 'rb') as handle:  
        data = pickle.load(handle)
    return data

cities = np.array(['barcelona', 'madrid', 'gijon', 'london', 'newyork', 'paris'])

for city in cities:

    datos = get_pickle("data/"+city+"/tripadimgrest_elvis_" + city + "/IMGMODEL/data_10+10/","TRAIN_DEV_IMG")
    df = pd.DataFrame(datos)

    # Eliminamos los negativos
    df = df[df['take'] != 0]

    # Creamos un nuevo pkl sin duplicados y sin negativos
    df.to_pickle("data/" + city + "/tripadimgrest_virgindata_" + city + "/" + city + "_DEV.pkl")
