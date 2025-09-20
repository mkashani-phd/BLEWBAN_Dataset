# %%
## you can run pip3 install -r requirements.txt to install all the packages
## but you need to install tensorflow or pytorch or keras manually
import pickle
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import tensorflow as tf
from keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, AveragePooling1D, Flatten, Dropout, Input,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

import cvnn.layers as complex_layers
import cvnn.activations 
import cvnn.losses as complex_losses







import  sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # try to use CPU only

# addin path to import IQ module
sys.path.append('../../')
import src.IQ as IQ

# %%
myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/admin")
# myclient = pymongo.MongoClient("mongodb://test:12345678910111213@SG-pine-beat-9444-57323.servers.mongodirector.com:27017/BLE")

BLE = myclient["BLE"]

def query(collection, filter:dict, addFrameColumn=True):
    df =  pd.DataFrame(list(collection.find(filter)))
    if addFrameColumn:
        df['frame'] = df.apply(lambda x: x['I'] + np.dot(x['Q'],1j), axis=1)
    return df.copy()

# %%
iq = IQ.IQ(Fc=2439810000+.1e4)

def configCreator(downSampleRate = 1, cutoff = 2e6):
    downSampleRate= max(downSampleRate, 1)
    return {                                      
            iq.butter:{'Fs': iq.Fs//downSampleRate, "cutoff": cutoff},
            iq.downSample:{'downSampleRate':downSampleRate, "shift": 0},
            iq.demodulate:{'Fs': iq.Fs},
           }

def freqCreator():
    return{
            iq.gradient:{},
            iq.unwrapPhase:{},
            iq.phase:{}, 
        }

# %%




def CV_CNN(X_train, y_train_encoded, X_test, y_test_encoded, downSampleRate=1):

    inputs = complex_layers.complex_input(shape=(X_train.shape[1],1,))
    x = complex_layers.ComplexConv1D(filters= 2, activation=cvnn.activations.cart_relu, kernel_size=max(5,128//downSampleRate))(inputs)
    x = complex_layers.ComplexAvgPooling1D(pool_size=2)(x)
    x = complex_layers.ComplexConv1D(filters= 2, activation=cvnn.activations.cart_relu, kernel_size=max(5,128//downSampleRate))(x)
    x = complex_layers.ComplexAvgPooling1D(pool_size=2)(x)
    x = complex_layers.ComplexDropout(0.1/downSampleRate)(x)
    x = complex_layers.ComplexFlatten()(x)
    # c1 = complex_layers.ComplexDense(64, activation='cart_relu', kernel_regularizer=regularizers.L1(0.001))(c1)
    x = complex_layers.ComplexDense(100, activation=cvnn.activations.convert_to_real_with_abs, kernel_regularizer=regularizers.L1(0.001))(x)
    x = Dense(100, activation='relu', kernel_regularizer=regularizers.L1(0.001))(x)
    out = Dense(y_test_encoded.shape[1], activation='softmax')(x)  # 13 classes

    model = tf.keras.Model(inputs, out)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    model.fit(X_train, y_train_encoded, epochs=512, batch_size=1024, validation_data=(X_test, y_test_encoded),verbose=0,callbacks=[early_stopping])
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    return accuracy

def RV_CNN(X_train, y_train_encoded, X_test, y_test_encoded, downSampleRate=1):
    input_shape = (2,X_train.shape[2])

    # Initialize the model
    model = Sequential()
    model.add(Conv1D(filters= 2, kernel_size=max(5,128//downSampleRate), activation='relu', input_shape=input_shape, data_format='channels_first', padding='same'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Conv1D(filters= 2, kernel_size=max(5,128//downSampleRate), activation='relu',data_format='channels_first', padding='same'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.1/downSampleRate))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(y_test_encoded.shape[1], activation='softmax'))  # Use 'softmax' for classification, 'linear' for regression
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Use 'mean_squared_error' for regression
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    model.fit(X_train, y_train_encoded, epochs=512, batch_size=1024, validation_data=(X_test, y_test_encoded),verbose=0,callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    return accuracy




def runForExperiment(df, target:str, mode:str ,query:dict, configurations:dict):
       
    for config in configurations.keys():
        found = False
        with open('results.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if f'{mode},{config},{query},{target}' in line:
                    print(f'Already done for {config}')
                    found = True
                    break
        if found:
            continue
        tf.keras.backend.clear_session()
        downSampleRate = list(configurations[config].values())[1]['downSampleRate']
        min_length = df['frame'].apply(len).min()
        min_length = min(min_length, 2000)// downSampleRate
        print(min_length)
        if mode == 'freq':
            temp = iq.apply(methods={**freqCreator(),**configurations[config]}, frame=df)
        else:
            temp = iq.apply(methods=configurations[config], frame=df)
        
        if mode == 'IQSplit':
            temp = temp.apply(lambda x: np.concatenate([np.real(x[0:min_length]), np.imag(x[0:min_length])]).reshape(2,-1))
        else:
            temp = temp.apply(lambda x: x[0:min_length])
        
        X_train, X_test, y_train, y_test = train_test_split(temp, df[target], test_size=0.2, random_state=22)

        X_train = tf.convert_to_tensor(X_train.tolist())
        X_test =  tf.convert_to_tensor(X_test.tolist())
        # y_train =  tf.convert_to_tensor(y_train.tolist())
        # y_test = tf.convert_to_tensor(y_test.tolist())

        y_train_encoded = to_categorical(y_train)
        y_test_encoded = to_categorical(y_test)
        # try:
        if mode == 'CV_NN':
            accuracy = CV_CNN(X_train, y_train_encoded, X_test, y_test_encoded, downSampleRate)
        else:
            accuracy = RV_CNN(X_train, y_train_encoded, X_test, y_test_encoded, downSampleRate)
        print(f'Accuracy for {config} is {accuracy}')
        with open('results.txt', 'a') as f:
            f.write(f'{mode},{config},{query},{target},{accuracy},{len(df)}\n')
            f.close()
        # except:
        #     print(f'Error for {config}')
    


# %%
configurations = { 
    '4,100': configCreator(downSampleRate = 1, cutoff=4e6), 
    '4,10': configCreator(downSampleRate=  10, cutoff=4e6), 
    
    
    '2,100': configCreator(downSampleRate = 1, cutoff=2e6), 
    '2,10': configCreator(downSampleRate=  10, cutoff=2e6), 
    '2,5': configCreator(downSampleRate = 20,  cutoff=2e6), 
    
    '1,100': configCreator(downSampleRate = 1, cutoff=1e6), 
    '1,10': configCreator(downSampleRate=  10, cutoff=1e6), 
    '1,5': configCreator(downSampleRate = 20, cutoff=1e6),
    '1,2.5': configCreator(downSampleRate = 40, cutoff=1e6),
}



queries = {


    # 'E1':{'test': 'offBody', 'query':{'SDR':'1', 'antenna': '1', 'txPower': '9dbm', 'pos': '4'}, 'target':'dvc'}, # done
    # 'E2':{'test':'offBody', 'query':{'SDR':'1', 'antenna': '1', 'txPower': '9dbm', 'dvc': '6'}, 'target':'pos'}, # done
    # 'E3':{'test':'offBody', 'query':{'antenna': '1', 'txPower': '9dbm', }, 'target':'pos'},# done
    # 'E4':{'test':'offBody', 'query':{'antenna': '1', 'txPower': '9dbm', }, 'target':'dvc'},# done
    # 'E5':{'test':'offBody', 'query':{'SDR': '1', 'txPower': '9dbm', }, 'target':'pos'},# done
    # 'E6':{'test':'offBody', 'query':{'SDR': '1', 'txPower': '9dbm',}, 'target':'dvc'},# done

    # 'E7':{'test':'onBody', 'query':{'pos':'static','antenna_side':'left'}, 'target':'dvc'},
    # 'E8':{'test':'onBody', 'query':{'pos':'static','antenna_side':'right'}, 'target':'dvc'},
    # 'E9':{'test':'onBody', 'query':{'pos':'moving','antenna_side':'right'}, 'target':'dvc'},# done
    # 'E10':{'test':'onBody', 'query':{'pos':'static'}, 'target':'dvc'}, # done
    # 'E11':{'test':'onBody', 'query':{'pos':'moving'}, 'target':'dvc'}, # done
    'E12':{'test':'onBody', 'query':{}, 'target':'dvc'}, # done
}


# %%
Modes = [
                # 'freq', 
                # 'IQSplit',
                'CV_NN'
                ]

for key in queries.keys():
    # try:
    print(key)
    df = query(BLE[queries[key]['test']], queries[key]['query'])

    # with open(f'data/{key}.pkl', 'rb') as f:
    #     df = pickle.load(f)
    # df = pd.DataFrame(df)
    # df['frame'] = df.apply(lambda x: x['I'] + np.dot(x['Q'],1j), axis=1)

    for mode in Modes:
        print(mode)
        runForExperiment(df =  df ,query=key, target= queries[key]['target'], mode= mode, configurations=configurations)
     # except:
        # sys.exit(0)

