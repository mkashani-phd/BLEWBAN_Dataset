# BLEWBAN_Dataset
BLEWBAN is a raw RF dataset of Bluetooth Low Energy (BLE) signals focused on Wireless Body Area Netwrok (WBAN). It consists of on-body and off-body recordings usign ESP32s in BLE mode.
The advantages of this dataset is:
- It covers the entire bandwith of the BLE technology.(recorded at 2.44GHz at 100Sps)
- Recording in anechoic chamber to reduce unwanted signals or interference.
- On-body recording on 12 different locations including: both left and right head, arm, wrist, chest, front and back torso (waist).  
- Off-body recording with the same devices at 7-different orientations
  
## How to access dataset
Python tools in this rpository provide a user-friendly access to the dataset stored in a **MongoDB database** on the cloud.
To use this dataset you don't need to download the raw files (22GB) and locally manage them. MongoDB database can easily run queries in multiple languages even on online platform such as google colab. Supported languages can be found [here](https://www.mongodb.com/languages).

Bellow is an example of how to perform a query and store the results in a Pandas data frame on a google colab. it can also be found in [here](https://colab.research.google.com/drive/1MDBT2rkZK7mvF0-5CpkBp85WYFNymxvO?usp=sharing)!

### Installing requierments
```python
!pip install -q pymongo
!pip install pandas
!pip install numpy
import pymongo
import numpy as np
import pandas as pd
```
### Making connection to MongoDB database
To connect to mongoDB database you need a username and a password. 

- username: **test**
- password: **12345678910111213**

there are two databases available.
- BLE
- BLE_metadata

BLE_metadata is the light version of the BLE excluding raw data. It contains basic time and frequency charectristics of the raw data and is much faster. On the other hand, The BLE database include raw recordings along with BLE_metadata.

For BLE_metadata use the following connection string
```python
client = pymongo.MongoClient("mongodb://test:12345678910111213@SG-pine-beat-9444-57323.servers.mongodirector.com:27017/BLE_metadata")
BLE_WBAN = client["BLE_metadata"]
```
Or for BLE use the following connection string
```python
client = pymongo.MongoClient("mongodb://test:12345678910111213@SG-pine-beat-9444-57323.servers.mongodirector.com:27017/BLE")
BLE_WBAN = client["BLE"]
```

### Quering data 
For example we are trying to filter off-body data recorded at atantenna 2 from device 1 in position 6 with 9dbm TX power.    
```python
filter = {'antenna': '2', 'dvc':'1', 'pos':'1', 'pos': '6', 'txPower': '9dbm'}
query = list(BLE_WBAN.offBody.find(filter))
df = pd.DataFrame(query)
```

