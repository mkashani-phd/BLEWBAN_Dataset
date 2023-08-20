# BLEWBAN_Dataset_Tool
BLEWBAN is a raw RF dataset recorded for healthcare sensor. It consists of on-body and off-body recordings usign ESP32s in BLE mode.
The advantages of this dataset is:
- It covers the entire bandwith of the BLE technology.(recorded at 2.44GHz at 100Sps)
- Recording in anechoic chamber to reduce unwanted signals or interference.
- on-body recording on 12 different locations including: both left and right head, arm, wrist, chest, front and back torso (waist)  
- off-body recording with the same devices at different orientation
## How to access dataset
Python tools in this rpository provide a user-friendly access to the dataset stored in a **MongoDB database** on the cloud.
To use this dataset you don't need to download the raw recording files and locally manage them. MongoDB database can easily run queries in multiple languages. Supported languages can be found [here](https://www.mongodb.com/languages).
