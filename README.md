# BLEWBAN_Dataset
BLEWBAN is a raw RF dataset of Bluetooth Low Energy (BLE) signals focused on Wireless Body Area Netwrok (WBAN). It consists of on-body and off-body recordings usign ESP32s in BLE mode.
The advantages of this dataset is:
- It covers the entire bandwith of the BLE technology.(recorded at 2.44GHz at 100MSps)
- Recording in anechoic chamber to reduce unwanted signals or interference.
- On-body recording on 12 different locations including: both left and right head, arm, wrist, chest, front and back torso (waist).  
- Off-body recording with the same devices at 7-different orientations
 
There is ongoing research on analyzing the on-body data. The plot bellow shows the classification results of **12 on-Body** devices in moving and static body positions.

![img](src/img/results.png)
  
## How to access dataset
Python tools in this repository provide a user-friendly access to the dataset stored in a **MongoDB database**.
To use this dataset you don't need to download the raw files (22GB) and locally manage them. MongoDB database can easily run queries in multiple languages even on online platforms such as Google Colab (example [here](https://colab.research.google.com/drive/1MDBT2rkZK7mvF0-5CpkBp85WYFNymxvO?usp=sharing)). Supported languages can be found [here](https://www.mongodb.com/languages). 

An example of how to perform a query and store the results in a Pandas data frame on a Google Colab is available !


### Installing requierments
``` 
pip3 install pymongo
pip3 install pandas
pip3 install numpy
```
After succesfull installation import dependencies.
```python
import pymongo
import numpy as np
import pandas as pd
```
### Making connection to MongoDB database
There are two databases available.
- BLE
- BLE_metadata
  
BLE_metadata is the light version of the BLE excluding raw data. It contains basic time and frequency charectristics of the raw data and is much faster. On the other hand, The BLE database include raw recordings along with BLE_metadata.
  
For **BLE** use the following connection string
```python
client = pymongo.MongoClient("mongodb://test:12345678910111213@SG-pine-beat-9444-57323.servers.mongodirector.com:27017/BLE")
BLE_WBAN = client["BLE"]
```

Or for **BLE_metadata** use the following connection string
```python
client = pymongo.MongoClient("mongodb://test:12345678910111213@SG-pine-beat-9444-57323.servers.mongodirector.com:27017/BLE_metadata")
BLE_WBAN = client["BLE_metadata"]
```

## Quering data 
For example we are trying to filter off-body data recorded at atantenna 2 from device 1 in position 6 with 9dbm TX power.    
```python
filter = {'antenna': '2', 'dvc':'1', 'pos': '6', 'txPower': '9dbm'}
query = list(BLE_WBAN.offBody.find(filter))
df = pd.DataFrame(query)
```
the above store a dataframe with columns such as:
|   |            _id           | frame_origin_file | frame_nr |     date    | dvc | pos | SDR |   test  | txPower | antenna | ... |     Fc     | gain |        frameTime       | lenFrame | frameChnl |       frameDecode       |       bitLen       |   max_gradient_unwrapped_phase   |                I               |                Q               |
|:-:|:------------------------:|:-----------------:|:--------:|:-----------:|:---:|:---:|:---:|:-------:|:-------:|:-------:|:---:|:----------:|:----:|:----------------------:|:--------:|:---------:|:-----------------------:|:------------------:|:--------------------------------:|:------------------------------:|:------------------------------:|
| 0 | 641aa1d333089480e2d0a5ca |   /home/moh/...   |     1    | jan 13 2023 |  1  |  6  |  1  | offBody |   9dbm  |    2    | ... | 2439810000 | 31.5 | [1478146.0, 1487348.0] |  9202.0  |     3     |  0x5553d3d3c468a54267b1 | [185, 92, 101, ... | [-0.014498871, 0.0127564659, ... |  [-0.01559495, -0.0057069, ... | [-0.00430310, 0.014526810, ... |
| 1 | 641aa1d333089480e2d0a5cb |   /home/moh/...   |     3    | jan 13 2023 |  1  |  6  |  1  | offBody |   9dbm  |    2    | ... | 2439810000 | 31.5 | [2978115.0, 2987351.0] |  9236.0  |     12    | 0x5553d3d3c4b4f71616de0 | [261, 96, 105, ... | [-0.013399868, 0.0109354282, ... |  [0.00198370, -0.0071718, ...  | [-0.01312295, -0.01129184, ... |
| 2 | 641aa1d333089480e2d0a5cc |   /home/moh/...   |     8    | jan 13 2023 |  1  |  6  |  1  | offBody |   9dbm  |    2    | ... | 2439810000 | 31.5 | [4478106.0, 4487332.0] |  9226.0  |     21    |  0x5553d3d3c40ce5d128c1 | [256, 96, 105, ... | [-0.013268690, 0.0109537563, ... | [-0.00790429, -0.00302133, ... | [-0.00939970, -0.01232947, ... |
| 3 | 641aa1d333089480e2d0a5cd |   /home/moh/...   |    10    | jan 13 2023 |  1  |  6  |  1  | offBody |   9dbm  |    2    | ... | 2439810000 | 31.5 | [5978104.0, 5987324.0] |  9220.0  |     30    | 0x5553d3d3c4f4645966200 | [251, 96, 104, ... | [-0.014260522, 0.0117406014, ... | [-0.00427259, -0.01013215, ... | [0.01345866, -0.005706963, ... |

## Data interpretaion
Since storing Complex numbers is not possible in the MongoDB database columns _I_ and _Q_ are real and imaginary numbers respectively. The code below will generate a new column named _frame_ that is $I + j*Q$
``` python
df['frame'] = query.apply(lambda x: x['I'] + np.dot(x['Q'],1j), axis=1)
```






