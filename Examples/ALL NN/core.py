# %%
## you can run pip3 install -r requirements.txt to install all the packages
## but you need to install tensorflow or pytorch or keras manually

import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import tensorflow as tf
from keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, Input


import cvnn.layers as complex_layers
import cvnn.losses as complex_losses







import threading, os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # try to use CPU only

# addin path to import IQ module
sys.path.append('../../')
import src.IQ as IQ

# %%
# myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/admin")
myclient = pymongo.MongoClient("mongodb://test:12345678910111213@SG-pine-beat-9444-57323.servers.mongodirector.com:27017/BLE")

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



def CVNN(X_train, y_train_encoded, X_test, y_test_encoded, downSampleRate=1):
    inputs = complex_layers.complex_input(shape=(X_train.shape[1],1,))
    c0 = complex_layers.ComplexConv1D(5, activation='cart_relu', kernel_size=max(5,100//downSampleRate))(inputs)
    c1 = complex_layers.ComplexConv1D(5, activation='cart_relu', kernel_size=max(5,100//downSampleRate))(c0)
    c1 = complex_layers.ComplexDropout(0.5/downSampleRate)(c1)
    c1 = complex_layers.ComplexFlatten()(c1)
    # c1 = complex_layers.ComplexDense(64, activation='cart_relu', kernel_regularizer=regularizers.L1(0.001))(c1)
    out = complex_layers.ComplexDense(100, activation='convert_to_real_with_abs', kernel_regularizer=regularizers.L1(0.001))(c1)
    out = Dense(100, activation='relu', kernel_regularizer=regularizers.L1(0.001))(out)
    out = Dense(y_test_encoded.shape[1], activation='softmax')(out)  # 13 classes

    model = tf.keras.Model(inputs, out)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train_encoded, epochs=128, batch_size=100, validation_data=(X_test, y_test_encoded),verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    return accuracy

def RVNN(X_train, y_train_encoded, X_test, y_test_encoded, downSampleRate=1):

    input_data = Input(shape=(X_train.shape[1],1,))
    x = Conv1D(filters=5, kernel_size=max(5,100//downSampleRate), activation='relu',)(input_data)
    x = Conv1D(filters=5, kernel_size=max(5,100//downSampleRate), activation='relu',)(x)
    x = Dropout(0.5/downSampleRate)(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu',kernel_regularizer=regularizers.L1(0.001))(x)
    x = Dense(100, activation='relu',kernel_regularizer=regularizers.L1(0.001))(x)
    output = Dense(y_test_encoded.shape[1], activation='softmax')(x) 

    model = Model(inputs=input_data, outputs=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train_encoded, epochs=128, batch_size=100, validation_data=(X_test, y_test_encoded),verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    return accuracy




def runForExperiment(df, target:str, mode:str ,query:dict, configurations:dict):
       
    for config in configurations.keys():
        found = False
        with open('reults.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if f' mode:{mode}, config:{config}, query:{query}, target: {target}' in line:
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
            temp = temp.apply(lambda x: np.concatenate([np.real(x[0:min_length]), np.imag(x[0:min_length])]))
        else:
            temp = temp.apply(lambda x: x[0:min_length])#cropping the input for complex input
        
        X_train, X_test, y_train, y_test = train_test_split(temp, df[target], test_size=0.2, random_state=42)

        X_train = tf.convert_to_tensor(X_train.tolist())
        X_test =  tf.convert_to_tensor(X_test.tolist())
        y_train =  tf.convert_to_tensor(y_train.tolist())
        y_test = tf.convert_to_tensor(y_test.tolist())

        y_train_encoded = to_categorical(y_train)
        y_test_encoded = to_categorical(y_test)
        try:
            if mode == 'CVNN':
                accuracy = CVNN(X_train, y_train_encoded, X_test, y_test_encoded, downSampleRate)
            else:
                accuracy = RVNN(X_train, y_train_encoded, X_test, y_test_encoded, downSampleRate)
            print(f'Accuracy for {config} is {accuracy}')
            with open('reults.txt', 'a') as f:
                f.write(f' mode:{mode}, config:{config}, query:{query}, target: {target}, accuracy:{accuracy}, #frames: {len(df)} \n')
                f.close()
        except:
            print(f'Error for {config}')
    


# %%
configurations = { 
    'butter4MHz_Fs100MHz': configCreator(downSampleRate = 1, cutoff=4e6), 
    'butter4MHz_Fs10MHz': configCreator(downSampleRate=  10, cutoff=4e6), 
    
    
    'butter2MHz_Fs100MHz': configCreator(downSampleRate = 1, cutoff=2e6), 
    'butter2MHz_Fs10MHz': configCreator(downSampleRate=  10, cutoff=2e6), 
    'butter2MHz_Fs5MHz': configCreator(downSampleRate = 20,  cutoff=2e6), 
    
    'butter1MHz,Fs100MHz': configCreator(downSampleRate = 1), 
    'butter1MHz_Fs10MHz': configCreator(downSampleRate=  10), 
    'butter1MHz_Fs5MHz': configCreator(downSampleRate = 20),
}



queries = {
    # 'E7':{'test':'onBody', 'query':{'pos':'static','antenna_side':'left'}, 'target':'dvc'},
    # 'E8':{'test':'onBody', 'query':{'pos':'static','antenna_side':'right'}, 'target':'dvc'},
    # 'E9':{'test':'onBody', 'query':{'pos':'moving','antenna_side':'right'}, 'target':'dvc'},# done
    # 'E10':{'test':'onBody', 'query':{'pos':'static'}, 'target':'dvc'}, # done
    # 'E11':{'test':'onBody', 'query':{'pos':'moving'}, 'target':'dvc'}, # done

    # 'E1':{'test': 'offBody', 'query':{'SDR':'1', 'antenna': '1', 'txPower': '9dbm', 'pos': '4'}, 'target':'dvc'}, # done
    # 'E2':{'test':'offBody', 'query':{'SDR':'1', 'antenna': '1', 'txPower': '9dbm', 'dvc': '6'}, 'target':'pos'}, # done
    # 'E3':{'test':'offBody', 'query':{'antenna': '1', 'txPower': '9dbm', }, 'target':'pos'},# done
    # 'E4':{'test':'offBody', 'query':{'antenna': '1', 'txPower': '9dbm', }, 'target':'dvc'},# done
    # 'E5':{'test':'offBody', 'query':{'SDR': '1', 'txPower': '9dbm', }, 'target':'pos'},# done
    # 'E6':{'test':'offBody', 'query':{'SDR': '1', 'txPower': '9dbm',}, 'target':'dvc'},# done
}


# %%
Experiments = [
                'freq', 
                'IQSplit',
                'CVNN'
                ]
for E in queries.keys():
    # try:
    print(queries[E]['query'])
    df = query(BLE[queries[E]['test']], queries[E]['query'])
    for experiment in Experiments:
        runForExperiment(df =  df ,query=E, target= queries[E]['target'], mode= experiment, configurations=configurations)
    # except:
        # sys.exit(0)


