# EXAMPLES
In this folder we are exploring the IQ package and demonstrate some classification examples using Deep Learning.

## BLEWBAN tutorial
The Analysis.ipynb has a thorough explanation of the BLEWBAN and the powerfull IQ package that is in this repository. You can find this code under [BLEWBAN_Tutorial](https://colab.research.google.com/drive/1MDBT2rkZK7mvF0-5CpkBp85WYFNymxvO?usp=drive_link) on Google colab where you can run the command online.

## CNN Classifier
For a quick start please check [CNN_BLEWBAN](https://colab.research.google.com/drive/1mY_gzbL6OIYSIrTMHnofLrjAYQz99-Es?usp=sharing) on Google colab.

## Anomaly detection usign Autoencoders
-- under progeress

## Machine learnign Classifier
To analyze deeper the hidden information of the data we have utilized 12 different machine learning algorithm on a proposed list of features.

### Inputs
We have tried many different subset of the dataset as input to the machine learning models. Since we divide the inputs to on-Body and off-Body main categories.

#### onBody
- static, antenna: left
- static, antenna: right
- moving,  antenna: left
- moving, antenna: right
- onBody, static
- onBody, moving
#### offBody
- offBody, antenna 1 SDR 1
- offBody, antenna 2 SDR 1

### Preprocessing
We have not used the RAW IQ as the input. We have used the list of method bellow to preprocess the IQ samples.
```python
return: {iq.bitFinderFromPhaseGradient:{'Fs': iq.Fs/downSampleRate},
        iq.scalePhaseGradientToHz: {'Fs': iq.Fs/downSampleRate}, 
        iq.gradient:{},
        iq.unwrapPhase:{},
        iq.phase:{}, 
        iq.butter:{'Fs': iq.Fs/downSampleRate, "cutoff": cutoff},
        iq.downSample:{'downSampleRate':downSampleRate, "shift": 0},
input:  iq.demodulate:{'Fs': iq.Fs}}
 ```

MinMax scaling normaliztion is also performed.
SMOTE is also an option, however we didn't use it for these results.

### Feature Extraction
After preprocessing the IQ samples we extract RSSI, Center Freq., firstBitLength, and [overshoot, STD, mean, len] for each freq. Deviation.

### PreProcessing effect?
Then we feed classification model under multiple differnet preprocessing parameters including downsampling rate and different butterworth cutoff frequency.

``` python
'butter4MHz_Fs100MHz': configCreator(downSampleRate = 1, cutoff=4e6), 
'butter4MHz_Fs10MHz': configCreator(downSampleRate=  10, cutoff=4e6), 


'butter2MHz_Fs100MHz': configCreator(downSampleRate = 1, cutoff=2e6), 
'butter2MHz_Fs10MHz': configCreator(downSampleRate=  10, cutoff=2e6), 
'butter2MHz_Fs5MHz': configCreator(downSampleRate = 20,  cutoff=2e6), 
 
'butter1MHz_Fs100MHz': configCreator(downSampleRate = 1), 
'butter1MHz_Fs10MHz': configCreator(downSampleRate=  10), 
'butter1MHz_Fs5MHz': configCreator(downSampleRate = 20), 
'butter1MHz_Fs2.5MHz': configCreator(downSampleRate = 40),
```
### ML models
The machine learning algorithm used includs:  GaussianNB, KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier, MLPClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier, HistGradientBoostingClassifier

### Results
this is a sample result

``` python
#butter1MHz_Fs2.5MHz 
{'SVM': 0.57, 'randomForest': 0.54, 'KNN': 0.53, 'Naive Bayse': 0.47, 'logreg': 0.48, 'DecisionTree': 0.56, 'AdaBoostClassifier': 0.27, 'GradientBoostingClassifier': 0.65, 'ExtraTreesClassifier': 0.66, 'BaggingClassifier': 0.68, 'HistGradientBoostingClassifier': 0.71, 'VotingClassifier': 0.62}
```


To see the full result please look at the MachineLearning.ipynb.
We are still working on vizualising the output.




# IQ package
IQ package is one of the best way to get started on this dataset. It is a powerful tool to do basic processing on the RAW IQ samples with a very user friendly interface. for more detail please check the analysis.ipynb

To start with the IQ package you first import the module form the src directory
Remember you might not need to add the path to sys.path
``` python
# Optional 
import sys
sys.path.append("..")  # Adds the parent directory to the sys.path

import src.IQ as IQ
```

Now you have to create
