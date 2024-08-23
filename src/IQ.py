import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from itertools import groupby
from operator import itemgetter
import scipy

from itertools import groupby
from operator import itemgetter

class IQ:
    Warnings = True
    Fc = 2.44e9
    Fs = 100e6
    figsize = (20,3)
    dpi = 100
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
    
    def inputCheck(self, input, method = None, col_name = None, args = None, plot = False):
        if input is None:
            if self.df is None:
                print("error: no input")
            else:
                input = self.df
        if method is None:
            print("error: no method")
            return
        

        if self.isList(input):
            if args is not None:
                return method(input, **args)
            else:
                return method(input)
        
        elif self.isPandaDF(input):
            if isinstance(input, pd.Series):
                if args is not None:
                    res = input.apply(lambda x: method(x,**args))
                else:
                    res = input.apply(lambda x: method(x))
                
            elif plot: # bad way to handle plot but this is a quick fix 
                try:  
                    res = input.apply(lambda x: method(x[col_name],x['title'],x['x_label'],x['y_label'], x['x']) , axis=1)
                except:
                    print("Warning: No x/y_label columns")
                    try:
                        res = input.apply(lambda x: method(x[col_name],x['title']) , axis=1)
                    except:
                        if self.Warnings:
                            print("Warning: Np title columns")
                        res = input.apply(lambda x: method(x[col_name],**args) , axis=1)

                return True 
            
            elif 'frame' in input.columns:
                if args is not None:
                    res = input.apply(lambda x: method(x['frame'],**args) , axis=1)
                else:
                    res = input.apply(lambda x: method(x['frame']) , axis=1)
            elif 'I' in input.columns and 'Q' in input.columns:
                res = input.apply(lambda x: method(x['I'] + np.dot(x['Q'],1j),**args) , axis=1)
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
    
    def _shift(self, input, shift = 0):
        return input * np.exp(2j*np.pi*shift*np.linspace(0,len(input),len(input))/len(input))
    def shift(self, frame: np.ndarray | pd.DataFrame = None, shift = 0, col_name = None):
        return self.inputCheck(frame, method=self._shift, col_name = col_name, args = {"shift": shift})
    
    def _rssi(self,input):
        input = input[100:-100]
        return 10*np.log(np.average(np.sqrt(np.imag(input)**2 + np.real(input)**2)))
    def rssi(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._rssi, col_name = col_name)

    def _channelDetection(self, input, Fs = Fs):
        fft = self.fft(input)
        absfft = np.abs(fft)
        n0 = np.where(absfft == np.max(absfft))[0][0] 
        f= np.arange(-self.Fs/2,Fs/2,Fs/len(absfft))
        c0 = f[n0] + self.Fc
        try:
            return np.where(abs(self.BLEChnls-c0) <1e6)[0][0]
        except:
            return -1  
        
    def channelDetection(self, frame: np.ndarray | pd.DataFrame = None, col_name = None, Fs = None):
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("Warning: (channelDetection) No sampling frequency specified, using default Fs of {}Msps.".format(Fs/1e6))
        return self.inputCheck(frame, method=self._channelDetection, col_name = col_name, args = {"Fs": Fs})

    def _demodulate(self, input, Fs = None):
        chnl = self._channelDetection(input, Fs = Fs)
        Fc = self.BLEChnls[chnl]
        diffFc = (self.Fc - Fc) / (Fs/len(input))
        return input * np.exp(2j*np.pi*diffFc*np.linspace(0,len(input),len(input))/len(input))
    
    def demodulate(self, frame: np.ndarray | pd.DataFrame = None, col_name = None, Fs = None):
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("Warning: (demodulate) No sampling frequency specified, using default Fs of {} MSps.".format(Fs/1e6))
        return self.inputCheck(frame, method=self._demodulate, col_name = col_name, args = {"Fs": Fs})
    
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
        phase = np.unwrap(input)
        return  phase
    def unwrapPhase(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._unwrapPhase, col_name = col_name)
    
    def _gradient(self, input):
        return np.gradient(input)
    def gradient(self, frame: np.ndarray | pd.DataFrame = None, col_name = None):
        return self.inputCheck(frame, method=self._gradient, col_name = col_name)
    
    def _sinc(self, input, length = 30):
        t= np.linspace(.1,1,length)  
        lpf = np.sinc(t)
        return np.convolve(input,lpf)
    
    def sinc(self, frame: np.ndarray | pd.Series = None, col_name = None, length = None):
        if length is None:
            length = 30
            if self.Warnings:
                print("Warning: (sinc) No filter length specified, using default length of {}".format(length))
        return self.inputCheck(frame, method=self._sinc, col_name = col_name, args={"length": length})
    
    def _butter(self,input,cutoff=1e6, Fs = Fs):
        fltr = scipy.signal.butter(30, cutoff, 'low', analog=False, output='sos',fs=Fs)
        return scipy.signal.sosfilt(fltr, input) 
    def butter(self, frame: np.ndarray | pd.Series = None, col_name = None, cutoff = None, Fs = None):
        if cutoff is None:
            cutoff = 1e6
            if self.Warnings:
                print("Warning: (butter) No filter cutoff specified, using default cutoff of {}MHz".format(cutoff/1e6))
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("Warning: (butter) No filter sampling frequency specified, using default Fs of {}Msps".format(Fs/1e6))
        return self.inputCheck(frame, method=self._butter, col_name = col_name, args={"cutoff": cutoff, "Fs": Fs})
    

    def _smooth(self, input, window_len=11,window='hanning'):
        """smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window on the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        input:
            window_len: the dimension of the smoothing window; 
                        should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 
                    'bartlett', 'blackman'
                    flat window will produce a moving average smoothing.
        output:
            the smoother FIR filter

        see also: 
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, 
        numpy.convolve scipy.signal.lfilter"""
            
        # if x.ndim != 1:
        #     raise ValueError( "smooth only accepts 1 dimension arrays.")

        # if x.size < window_len:
        #     raise ValueError( "Input vector needs to be bigger than window size.")
        
        # if window_len<3:
        #     return x
        
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError( f"Window is on of '{'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}'")
        
        # s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval( f"np.{window}(window_len)")
        
        lpf = w/w.sum()
        res = np.convolve(input,lpf)
        return res[int(len(lpf)/2-1):-int(len(lpf)/2)]
    
    def smooth(self, frame: np.ndarray | pd.Series = None, col_name = None, window_len = 11, window = 'hanning'):
        return self.inputCheck(frame, method=self._smooth, col_name = col_name, args={"window_len": window_len, "window": window})
    
    def _downSample(self, input, downSampleRate =2, shift = 0):
        return input[shift:: downSampleRate]
    def downSample(self, frame: np.ndarray | pd.DataFrame = None, downSampleRate = 2, shift = 0, col_name = None):
        return self.inputCheck(frame, method=self._downSample, col_name = col_name, args = {"downSampleRate": downSampleRate, "shift": shift})
    
    def _upSample(self, input, upSampleRate =2):
        return np.repeat(input, upSampleRate)
    def upSample(self, frame: np.ndarray | pd.DataFrame = None, upSampleRate = 2, col_name = None):
        return self.inputCheck(frame, method=self._upSample, col_name = col_name, args = {"upSampleRate": upSampleRate})
    
    def _scalePhaseGradientToHz(self, input, Fs = Fs):
        return input * Fs / (2*np.pi)
    def scalePhaseGradientToHz(self, frame: np.ndarray | pd.DataFrame = None, col_name = None, Fs = None):
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("Important Warning: (scalePhaseGradientToHz) No sampling frequency specified, using default Fs of {}Msps.".format(Fs/1e6))
        return self.inputCheck(frame, method=self._scalePhaseGradientToHz, col_name = col_name, args = {"Fs": Fs})

    def keepPositive(self, samples):
        sam = samples.copy()
        sam[sam<0] = 0
        return sam
    def keepNegative(self, samples):
        sam = samples.copy()
        sam[sam>0] = 0
        return sam


    def _bitMetaDataGenerator(self, sample , indx, bitsPerSample):
        return [
                {
                    'samples': sample[x[0]:x[1]],
                    'numberOfBits':round((x[1] - x[0])/bitsPerSample), 
                    'indxBegining': x[0], 'indxEnd': x[1], 
                    'len': x[1] - x[0], 'slope':1,
                    'overshoot': np.max(np.abs(sample[x[0]:x[1]])),
                    # 'undershoot': np.min(np.abs(sample[x[0]:x[1]])),
                    'std': np.std(sample[x[0]:x[1]]),
                    'mean': np.mean(np.abs(sample[x[0]:x[1]])),
                } 
                for x in indx if np.max(np.abs(sample[x[0]:x[1]])) > 60000 ## will remove the scaled phase gradient bits that are not bits  
               ]


    def nonZeroGrouper(self, samples,biggerThan = None, smallerThan = None ,Fs = Fs, noGroupBefore = None):
        if smallerThan is None:
            smallerThan = 10000*Fs/self.Fs
        if biggerThan is None:
            biggerThan = 10*Fs/self.Fs
        if noGroupBefore is None:
            noGroupBefore = 0*Fs/self.Fs
            if self.Warnings:
                print("Warning: (nonZeroGrouper) No noGroupBefore specified, using default noGroupBefore of {}".format(noGroupBefore) )

        test_list = np.nonzero(samples)
        framesIndex = []
        for k, g in groupby(enumerate(test_list[0]), lambda ix: ix[0]-ix[1]):
            temp = list(map(itemgetter(1), g))
            if len(temp)< biggerThan or len(temp)> smallerThan:
                continue
            if temp[0] > noGroupBefore: # no bits befor 1100 samples at 100e6 sample rate
                framesIndex.append([temp[0],temp[-1]])
        return np.array(framesIndex)



    # the default values are for 100Msps
    # the BLE symbol rate is 1 million symbol per second. 
    def _bitFinderFromPhaseGradient(self, sample, Fs= Fs, bitsPerSample = None, biggerThan = None, smallerThan = None , noGroupBefore = None, plot = False, col_name = None, title = None, x_label = None, y_label = None):
        if bitsPerSample is None:
            bitsPerSample = Fs/1e6
            if self.Warnings:
                print("Warning: (bitFinderFromPhaseGradient) No bits per sample specified, using default bitsPerSample of {}".format(bitsPerSample) )
        if biggerThan is None:
            biggerThan = int(0.82*bitsPerSample)
            if self.Warnings:
                print("Warning: (bitFinderFromPhaseGradient) No frame bigger than specified, using default biggerThan of {}".format(biggerThan) )
        if smallerThan is None:
            smallerThan = int(100*bitsPerSample)
            if self.Warnings:
                print("Warning: (bitFinderFromPhaseGradient) No frame smaller than specified, using default smallerThan of {}".format(smallerThan) )
        
        X_positive = self.keepPositive(sample)
        X_negative = self.keepNegative(sample)
        pIndx = self.nonZeroGrouper(X_positive, Fs=Fs, biggerThan = biggerThan, smallerThan = smallerThan, noGroupBefore = noGroupBefore)
        nIndx = self.nonZeroGrouper(X_negative, Fs=Fs, biggerThan = biggerThan, smallerThan = smallerThan, noGroupBefore = noGroupBefore)
        pIndx_meta = self._bitMetaDataGenerator(sample, pIndx, bitsPerSample)
        nIndx_meta = self._bitMetaDataGenerator(sample, nIndx, bitsPerSample)
    
        if plot:
            plt.figure(figsize=(20,3), dpi=100)
            plt.plot(np.zeros(max(len(X_positive),len(X_negative))))
            plt.plot(sample)
            plt.stem(pIndx.flatten(), [.3*np.max(X_positive)]*len(pIndx.flatten()) ,'r')
            plt.stem(nIndx.flatten(), [.3*np.min(X_negative)]*len(nIndx.flatten()))
            plt.title(title)
            plt.xlabel(x_label if x_label is not None else 'Samples')
            plt.ylabel(y_label if y_label is not None else 'Freq. Deviation (Hz)')
            plt.show()
            plt.close()
        return pd.DataFrame(sorted(pIndx_meta + nIndx_meta, key=lambda x: x["indxBegining"]))
    
    def bitFinderFromPhaseGradient(self, frame: np.ndarray | pd.Series = None, col_name = None, Fs = None, bitsPerSample = None, biggerThan = None, smallerThan = None, noGroupBefore = None,plot = False, title = None, x_label = None, y_label = None):
        if Fs is None:
            Fs = self.Fs
            if self.Warnings:
                print("IMPORTANT WARNING: (bitFinderFromPhaseGradient) No sampling frequency specified, using default Fs of {}Msps.".format(Fs/1e6))
        return self.inputCheck(frame, method=self._bitFinderFromPhaseGradient, col_name = col_name, args = {"Fs": Fs, "bitsPerSample": bitsPerSample, "noGroupBefore":noGroupBefore,  "biggerThan": biggerThan, "smallerThan": smallerThan, "plot": plot, "title": title, "x_label": x_label, "y_label": y_label})

    
    

    def _plotUtills(self, input, title = None, x_label = None, y_label = None,x = None, xscale = None):
        plt.figure(figsize=self.figsize,dpi=self.dpi)

        if x is not None:
            plt.plot(np.linspace(x[0],x[1], len(input)),input)
        else:
            plt.plot(input)

        if xscale is not None:
            plt.xscale('symlog', linthreshx=xscale)

        if title is not None:
            plt.title(title)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        plt.show()

    def _plot(self, input, title = None, x_label = None, y_label = None, x = None, xscale = None):
        if isinstance(input, pd.Series):
            for column in input:
                self._plotUtills(input=column, title = title, x_label = x_label, y_label = y_label, x = x,  xscale = xscale)
        else:
            self._plotUtills(input=input, title = title, x_label = x_label, y_label = y_label, x = x,  xscale = xscale)
        
    def plot(self, frame: np.ndarray | pd.Series | pd.DataFrame = None, col_name: str | list  = None, title: str  = None, x_label : str  = None, y_label: str = None, x: np.ndarray = None, xscale = None):
        args={'title': title, 'x_label': x_label, 'y_label': y_label, 'x': x , 'xscale': xscale}
        self.inputCheck(frame, method=self._plot, col_name = col_name,  args= args, plot = True)  


    # def _apply(self,  method, input = None,col_name = None,args = None):
    #     print(args)
    #     if args is not None:
    #         return method(input, col_name, **args)
    #     else:
    #         return method(input, col_name)

    
    def apply(self, methods: list| dict, frame: np.ndarray | pd.Series | pd.DataFrame = None, col_name: str | list  = None):
        if isinstance(methods, dict):
            method_keys = list(methods.keys())
            while len(method_keys) > 0:
                method_nm = method_keys.pop()
                if isinstance(method_nm, str):
                    method = self.__getattribute__(method_nm)
                else:
                    method = method_nm
                try:
                    method.__qualname__.startswith('IQ.')
                except:
                    frame = self.inputCheck(frame, method=method, col_name = col_name, args = methods[method_nm])
                    continue
                if not method.__qualname__.startswith('IQ.'): #User defined function
                    frame = self.inputCheck(frame, method=method, col_name = col_name, args = methods[method_nm])
                    continue
            
                if methods[method_nm] is not None: # if args is not None
                    try:
                        frame = method(frame = frame, col_name = col_name, **methods[method_nm])
                    except:
                        if self.Warnings:
                            print("**** Warning: args not applied ****")
                        frame = method(frame = frame, col_name = col_name)
                else:
                    frame = method(frame = frame, col_name = col_name)
                
        elif isinstance(methods, list):
            while len(methods) > 0:
                method_nm = methods.pop()
                if isinstance(method_nm, str):
                    method = self.__getattribute__(method_nm)
                else:
                    method = method_nm

                try:
                    method.__qualname__.startswith('IQ.')
                except:
                    frame = self.inputCheck(frame, method=method, col_name = col_name)
                    continue
                if not method.__qualname__.startswith('IQ.'): #User defined function
                    frame = self.inputCheck(frame, method=method, col_name = col_name)
                    continue
                frame = method(frame = frame, col_name = col_name)

        return frame



    

        
         