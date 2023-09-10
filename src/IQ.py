import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from itertools import groupby
from operator import itemgetter

class IQ:
    Warnings = True
    Fc = 2.44e9
    Fs = 100e6
    df = None
    BLEChnls = np.array([2404000000,2406000000,2408000000,2410000000,
    2412000000,2414000000,2416000000,2418000000,2420000000,2422000000,
    2424000000,2428000000,2430000000,2432000000,2434000000,2436000000,
    2438000000,2440000000,2442000000,2444000000,2446000000,2448000000,      
    2450000000,2452000000,2454000000,2456000000,2458000000,2460000000,
    2462000000,2464000000,2466000000,2468000000,2470000000,2472000000,
    2474000000,2476000000,2478000000,2402000000,2426000000,2480000000])

    onBodyMap = {1: ['head','right'],              2: ['head','left'], 
                  3: ['chest', 'right'],            4: ['chest', 'left'],
                  5: ['fornTorso', 'right'],        6: ['fornTorso', 'left'],
                  7: ['arm', 'right'],              8: ['arm', 'left'],
                  9: ['wrist', 'right'],           10: ['wrist', 'left'],
                  11: ['backTorso', 'right'],      12: ['backTorso', 'left']}
    
    def __init__(self, df = None, Fc = None, Fs = None, Warnings = True):
        if Fs is not None:
            self.Fs = Fs
        if Fc is not None:
            self.Fc = Fc
        if df is not None: 
            self.df = df
        self.Warnings = Warnings

    def isList(self, input):
        return isinstance(input, list) or isinstance(input,np.ndarray) 
    
    def isPandaDF(self, input):
        return isinstance(input, pd.DataFrame) or isinstance(input, pd.Series)
    
    def inputCheck(self, input, method = None, col_name = None, title = None, plot = False):
        if input is None:
            if self.df is None:
                print("error: no input")
            else:
                input = self.df
        if method is None:
            print("error: no method")
            return
        

        if self.isList(input):
            return method(input)
        
        elif self.isPandaDF(input):
            if isinstance(input, pd.Series):
                res = input.apply(lambda x: method(x))
                
            elif plot: # bad way to handle plot but this is a quick fix
                if title:  
                    try:  
                        res = input.apply(lambda x: method(x[col_name],x['title'],x['x_label'],x['y_label']) , axis=1)
                    except:
                        try:
                            res = input.apply(lambda x: method(x[col_name],x['title']) , axis=1)
                        except:
                            if self.Warnings:
                                print("Warning: input does not contain title or x/y_label columns")
                            res = input.apply(lambda x: method(x[col_name]) , axis=1)
                else:
                    res = input.apply(lambda x: method(x[col_name]) , axis=1)
                return True 
            
            elif 'frame' in input.columns:
                res = input.apply(lambda x: method(x['frame']) , axis=1)
            elif 'I' in input.columns and 'Q' in input.columns:
                res = input.apply(lambda x: method(x['I'] + np.dot(x['Q'],1j)) , axis=1)
            else:  
                print("error: input does not contain frame or I/Q columns")

            if col_name is not None:
                self.df[col_name] = res
                return self.df
            else:
                return res


    def _abs(self, input):
        return np.abs(input)
    def abs(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._abs, col_name = col_name)
    
    def _phase(self, input):
        return np.angle(input)
    def phase(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._phase, col_name = col_name)
    
    def _fft(self,input):
        return np.fft.fftshift(np.fft.fft(input))
    def fft(self,frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame,method = self._fft, col_name = col_name)
    
    def _rssi(self,input):
        input = input[100:-100]
        return 10*np.log(np.average(np.sqrt(np.imag(input)**2 + np.real(input)**2)))
    def rssi(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._rssi, col_name = col_name)

    def _channelDetection(self, input):
        fft = self.fft(input)
        absfft = np.abs(fft)
        n0 = np.where(absfft == np.max(absfft))[0][0] 
        f= np.arange(-self.Fs/2,self.Fs/2,self.Fs/len(absfft))
        c0 = f[n0] + self.Fc
        try:
            return np.where(abs(self.BLEChnls-c0) <1e6)[0][0]
        except:
            return -1  
    def channelDetection(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._channelDetection, col_name = col_name)

    def _demodulate(self, input):
        chnl = self.channelDetection(input)
        Fc = self.BLEChnls[chnl]
        diffFc = (self.Fc - Fc) / (self.Fs/len(input))
        return input * np.exp(2j*np.pi*diffFc*np.linspace(0,len(input),len(input))/len(input))
    def demodulate(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._demodulate, col_name = col_name)
    
    def _removeDC(self, input):
        return input - np.average(input)
    def removeDC(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._removeDC, col_name = col_name)
    

    
    
    #Github Copilot wrote this function, not sure if it works!
    def _findPeaks(self, input): 
        return np.where(np.diff(np.sign(np.diff(input))))[0] + 1
    def findPeaks(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._findPeaks, col_name = col_name)
    
    def _reconstruct(self, input):
        cos = np.real(input)*np.sin(2*np.pi* self.Fc * np.linspace(1,len(input),len(input))/self.Fs)
        sin = np.imag(input)*np.cos(2*np.pi* self.Fc * np.linspace(1,len(input),len(input))/self.Fs)
        return cos + sin
        # return np.abs(input)*np.cos(2*np.pi* self.Fc * np.linspace(0,len(input),len(input))/self.Fs + np.angle(input))
    def reconstruct(self,frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._reconstruct, col_name = col_name)
    
    def _unwrapPhase(self, input):
        phase = np.unwrap(np.angle(input))
        return  phase
    def unwrapPhase(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._unwrapPhase, col_name = col_name)
    
    def _gradient(self, input):
        return np.gradient(input)
    def gradient(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._gradient, col_name = col_name)
    
    def _sincFilter(self, input):
        t= np.linspace(.01,1,30)  
        lpf = np.sin(t)/t
        return np.convolve(input,lpf)
    def filter(self, frame: np.ndarray | pd.Series = None, filter = None, col_name = None):
        if filter is None:
            filter = self._sincFilter
            if self.Warnings:
                print("Warning: No filter specified, using sinc filter")
        return self.inputCheck(frame, method=filter, col_name = col_name)
    


    def _plotUtills(self, input, title = None, x_label = None, y_label = None):
        plt.figure(figsize=(20,3))
        plt.plot(input)
        if title is not None:
            plt.title(title)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        plt.show()

    def _plot(self, input, title = None, x_label = None, y_label = None):
        if isinstance(input, pd.Series):
            for column in input:
                self._plotUtills(input=column, title = title, x_label = x_label, y_label = y_label)
        else:
            self._plotUtills(input=input, title = title, x_label = x_label, y_label = y_label)
        
    def plot(self, frame: np.ndarray | pd.Series | pd.DataFrame = None, col_names: str | list  = None, title: bool = False):
        self.inputCheck(frame, method=self._plot, col_name = col_names, title = title, plot = True)  

    
    
    
    

        
         