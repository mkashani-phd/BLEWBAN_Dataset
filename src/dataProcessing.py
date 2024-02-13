import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import groupby
from operator import itemgetter



class IQdata:
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






    # SDR_1 / iter_1 / antenna 1 -> left   0
    # SDR_1 / iter_1 / antenna 2 -> right  1
    # |
    # V
    # SDR_1 / iter_2 / antenna 1 -> right  1
    # SDR_1 / iter_2 / antenna 2 -> left   0
    # |
    # V   
    # SDR_2 / iter_1 / antenna 1 -> right  1
    # SDR_2 / iter_1 / antenna 2 -> left   0
    # |
    # V
    # SDR_2 / iter_2 / antenna 1 -> left   0
    # SDR_2 / iter_2 / antenna 2 -> right  1


    # !(abc) + !a bc + a !b c + ab !c


    SERVER_RSSI = 0
    path = None
    samples = None
    TotalFramesIndex = None
    tindx = None
    LEN = 0
 

    def __init__(self, path, samples,tindx,Fc = None, Fs = None):
        self.path = path
        self.samples = samples
        self.TotalFramesIndex = self.frameFinder(samples)
        self.LEN = len(self.TotalFramesIndex)
        if Fs is None:
            self.Fs = 100e6
        else:
            self.Fs = Fs
        if Fc is None:
            self.Fc = 2.444e9
        else:
            self.Fc = Fc

        self.tindx = tindx.reshape(-1,2)

        [i for i in range(self.LEN) if self.channelDetection(i) in [37,38,39]]

    def isList(self, input):
        return isinstance(input, list) or isinstance(input,np.ndarray)
    
    def inputCheck(self, input):
        if self.isList(input):
            return input
        else:
            return self.frameByNumber(input)
             

    def frameFinder(self,samples,farmeBiggerThan = 150):
        test_list = np.nonzero(samples)
        framesIndex = []
        for k, g in groupby(enumerate(test_list[0]), lambda ix: ix[0]-ix[1]):
            temp = list(map(itemgetter(1), g))
            if len(temp)< farmeBiggerThan:
                continue
            framesIndex.append([temp[0],temp[-1]])
        return np.array(framesIndex)

    def indexByNumber(self,num):
        return self.TotalFramesIndex[num]

    def frameByIndex(self,index):
        return self.samples[index[0]:index[1]]

    def frameByNumber(self,frame_nr:int):
        return self.frameByIndex(self.TotalFramesIndex[frame_nr])
    
    def lenFrame(self,frame_nr:int):
        frame = self.inputCheck(frame_nr)
        return len(frame)

    def fft(self,frame_nr:int | np.ndarray):
        frame = self.inputCheck(frame_nr)
        return np.fft.fftshift(np.fft.fft(frame))
    
    def rssi(self, frame_nr):
        frame = self.inputCheck(frame_nr)[100:-100]
        return 10*np.log(np.average(np.sqrt(np.imag(frame)**2 + np.real(frame)**2)))  

    def channelDetection(self, frame_nr:int | np.ndarray):
        frame = self.inputCheck(frame_nr)
        fft = self.fft(frame)
        absfft = np.abs(fft)
        n0 = np.where(absfft == np.max(absfft))[0][0] 
        f= np.arange(-self.Fs/2,self.Fs/2,self.Fs/len(absfft))
        c0 = f[n0] + self.Fc
        try:
            return np.where(abs(self.BLEChnls-c0) <1e6)[0][0]
        except:
            return -1
    def isServer(self, frame_nr): # return true if frame is server frame or Unkonwn frame
        frame = self.inputCheck(frame_nr)
        try:
            frame_ahead = self.inputCheck(frame_nr+1)
        except:
            return True
        
        # if frame is channel 37,38,39 (or) channel is different from next frame then it is server frame
        if self.channelDetection(frame) in [37,38,39]:
            return True
        
        if self.channelDetection(frame) != self.channelDetection(frame_ahead):
            return True

        return False

    def getMetaData(self, frame_nr: int, include_frame = False):
        metaData = {}
        metaData['frame_origin_file'] = self.path
        metaData['frame_nr'] = frame_nr

        # SDR_1              offBody          high-tx-PWR-9dbm     ['x310-jan-13-dvc-1-pos-2', '2440000000.0', '100000000.0', '31.5', '2', '0', '2.iq']
    #split('/')[-4]      split('/')[-3]       split('/')[-2]       split('/')[-1].split('_')        
        # SDR_1              onBody                 iter_1         ['x310-jan-14-HighPower-dvc-3-pos-moving', '2440000000.0', '100000000.0', '31.5', '2', '1', '1.iq']

        temp = self.path.split('/')[-1].split('_')[0].split('-')
        #'x310-jan-13-dvc-1-pos-2'
        metaData['date'] = temp[1] + ' ' + temp[2] + ' 2023'

        metaData['SDR'] = self.path.split('/')[-4][-1]
        metaData['test'] = self.path.split('/')[-3]
        metaData['antenna'] = self.path.split('/')[-1].split('_')[-1][0]

        if metaData['test'] == "offBody":
            metaData['dvc'] = temp[4]
            metaData['pos'] = temp[6]
            metaData['txPower'] = self.path.split('/')[-2].split('-')[-1]
            
        elif metaData['test'] == "onBody":
            metaData['dvc'] = temp[5]
            metaData['txPower'] = '9dbm'

            metaData['placement'] = self.onBodyMap[int(temp[5])][0]
            metaData['side'] = self.onBodyMap[int(temp[5])][1]
            metaData['pos'] = temp[7]
            # SDR_1 / iter_1 / antenna 1 -> left   0
            # SDR_1 / iter_1 / antenna 2 -> right  1
            # |
            # V
            # SDR_1 / iter_2 / antenna 1 -> right  1
            # SDR_1 / iter_2 / antenna 2 -> left   0
            # |
            # V   
            # SDR_2 / iter_1 / antenna 1 -> right  1
            # SDR_2 / iter_1 / antenna 2 -> left   0
            # |
            # V
            # SDR_2 / iter_2 / antenna 1 -> left   0
            # SDR_2 / iter_2 / antenna 2 -> right  1

            # !(abc) + !a bc + a !b c + ab !c // wrong formula
            # ~a~bc + ~ab~c + a~b~c + abc
            a = int(metaData['SDR']) - 1
            b = int(self.path.split('/')[-2][-1]) -1
            c = int(self.path.split('/')[-1].split('_')[-1][0]) - 1
            print(a,b,c,((not a) and (not b) and c) or ( (not a) and b and (not c)) or (a and (not b) and (not c)) or (a and b and c))
            if ((not a) and (not b) and c) or ( (not a) and b and (not c)) or (a and (not b) and (not c)) or (a and b and c):
                metaData['antenna_side'] = 'right'
            else:
                metaData['antenna_side'] = 'left'
            


        metaData['Fs'] = self.Fs
        metaData['Fc'] = self.Fc
        metaData['gain'] = self.path.split('/')[-1].split('_')[3]
        metaData['frameTime'] = self.tindx[frame_nr].tolist()
        if frame_nr != 0:
            metaData['previousFrameTime'] = self.tindx[frame_nr-1].tolist()
        else:
            metaData['previousFrameTime'] = [-1,-1]
        if frame_nr != self.LEN - 1:
            metaData['nextFrameTime'] = self.tindx[frame_nr+1].tolist()
            metaData['timeTillNextFrame'] = int(self.tindx[frame_nr+1][0] - self.tindx[frame_nr][1])

        else:
            metaData['nextFrameTime'] = [-1,-1]
            metaData['timeTillNextFrame'] = -1
        metaData['lenFrame'] = self.tindx[frame_nr][1] - self.tindx[frame_nr][0]
        metaData['frameChnl'] = int(self.channelDetection(frame_nr))
        metaData['rssi'] = self.rssi(frame_nr)

        
        try:
            decoded = self.decode(frame_nr, lpf = self.smooth(window_len=60,window='flat')) 

            metaData['frameDecode'] = decoded[1]
            metaData['bitLen'] = decoded[2].tolist()
            metaData['max_gradient_unwrapped_phase'] = decoded[3].tolist()
        except:
            metaData['frameDecode'] = []
            metaData['bitLen'] = []
            metaData['max_gradient_unwrapped_phase'] = []

        if include_frame:
            metaData['I'] = np.real(self.frameByNumber(frame_nr)).tolist()
            metaData['Q'] = np.imag(self.frameByNumber(frame_nr)).tolist()


        return metaData
    
    def demodulate(self,frame_nr:int | np.ndarray):
        frame = self.inputCheck(frame_nr)
        chnl = self.channelDetection(frame)
        Fc = self.BLEChnls[chnl]
        diffFc = (self.Fc - Fc) / (self.Fs/len(frame))
        demod = frame * np.exp(2j*np.pi*diffFc*np.linspace(0,len(frame),len(frame))/len(frame))
        
        return demod


    def reconstructor(self,frame_nr:int | np.ndarray,Fc = None, Fs = None):
        frame = self.inputCheck(frame_nr)
        if Fc is None:
            Fc = self.Fc
        if Fs is None:
            Fs = self.Fs
        cos = np.real(frame)*np.sin(2*np.pi* Fc * np.linspace(0,len(frame),len(frame))/Fs)
        sin = np.imag(frame)*np.cos(2*np.pi* Fc * np.linspace(0,len(frame),len(frame))/Fs)
        return cos + sin

    def phase(self,frame_nr:int | np.ndarray):
        frame = self.inputCheck(frame_nr)
        return np.angle(frame)
    
    def unwrapPhase(self,frame_nr:int | np.ndarray):
        frame = self.inputCheck(frame_nr)
        return np.unwrap(frame)
    
    def demodAndPhase(self,  frm_nr, interval = [0,-1], chnl = -1):
        frame = self.inputCheck(frm_nr)
        if self.channelDetection(frm_nr) != chnl and chnl != -1:
            return 0, 0
        demod = self.demodulate(frame)
        phase = np.unwrap(np.angle(demod))
        return demod[interval[0]:interval[1]],phase[interval[0]:interval[1]]
    
    def smooth(self, window_len=11,window='hanning'):
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
        return w/w.sum()
    

    def filter(self,frame_nr:int | np.ndarray,window_len=11,window='hanning'):
        frame = self.inputCheck(frame_nr)
        w = self.smooth(window_len,window)
        return np.convolve(frame,w,mode='same')
            

    def decode(self, frm_nr,signal = [1000,-20], bitSamplePeriod = 92,lpf = None,plot = False, title = None):
        frame = self.inputCheck(frm_nr)
        demod = self.demodulate(frame)
        ### low pass filtering ##
        if lpf is None: 
            t= np.linspace(.01,1,100) # has to be automated eventually 
            lpf = np.sin(t)/t
        demod = demod[signal[0]:signal[1]]
        res = np.convolve(demod,lpf)
        res = res[int(len(lpf)/2-1):-int(len(lpf)/2)]

        phi = np.unwrap(np.angle(res))

        #adding zero to begining and end
        xpx = np.gradient(phi)
        xpx = np.append(xpx,[0,0,0])
        xpx = np.insert(xpx, 0 , 0)
        
        xnx = np.gradient(phi)
        xnx = np.append(xpx,[0,0,0])
        xnx = np.insert(xnx, 0 , 0)

        xpx[xpx<0] = 0
        xnx[xnx>0] = 0

        pIndx = self.frameFinder(samples=xpx,farmeBiggerThan=30)
        nIndx = self.frameFinder(samples=xnx,farmeBiggerThan=30)
        pDecode = []
        nDecode = []
        pLen = [] # length of the 1 bits
        pMax = [] # max value of the 1 bits
        nLen = [] # length of the 0 bits
        nMax = [] # max value of the 0 bits

        bitLen = []
        for i,j in pIndx:
            pDecode.append((j-i + 1) // bitSamplePeriod)
            pLen.append(j-i+1)
            pMax.append(np.max(xpx[i:j]))
        for i,j in nIndx:
            nDecode.append((j-i + 1) // bitSamplePeriod)
            nLen.append(j-i+1)
            nMax.append(np.min(xnx[i:j]))


        if len(pIndx) < 1 or len(nIndx) < 1: # if there is  no bit in the frame
            print(frm_nr)
            return np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1)
        
        max_gradient_unwrapped_phase = []
        if pIndx[0][0] < nIndx[0][0]:
            #zip order -> zip(p,n)
            decoded = list(zip(pDecode,nDecode))
            bitLen = list(zip(pLen,nLen))
            max_gradient_unwrapped_phase = list(zip(pMax,nMax))
            startingBit = 1
        else:
            #zip order -> zip(n,p)
            decoded = list(zip(nDecode,pDecode))
            bitLen = list(zip(nLen,pLen))
            max_gradient_unwrapped_phase = list(zip(nMax,pMax))
            startingBit = 0
        
        # zip doesn't add the unmatched part of the array with differnet sizes
        if len(pIndx) > len(nIndx):
            decoded.append([pDecode[-1],0])
            bitLen.append([pLen[-1],0])
            max_gradient_unwrapped_phase.append([pMax[-1],0])
        elif len(pIndx) < len(nIndx):
            decoded.append([nDecode[-1],0])
            bitLen.append([nLen[-1],0])
            max_gradient_unwrapped_phase.append([nMax[-1],0])

        bitLen = np.array(bitLen).flatten()
        decoded = np.array(decoded).flatten()
        max_gradient_unwrapped_phase = np.array(max_gradient_unwrapped_phase).flatten()

        decoded[0] = 1
        res = []
        for bit in decoded:
            for kk in range(bit):
                res.append(int(startingBit))
            startingBit = not startingBit
        
        binary = ''.join(map(str,res))

        resInHex = "0x"
        for i in range(len(binary)//4 + 1):
            try:
                resInHex += hex(int(binary[4*i:4*i+4],2))[-1]
            except:
                continue
        
        if plot:
    # def decodePlot(phi,xpx,xnx,pIndx,nIndx,bitSmaplePeriod):
            plt.figure(figsize=(20, 3), dpi=1000)
            # plt.plot(phi/200)
            plt.plot(np.zeros(len(phi)))
            plt.plot(xpx)
            plt.plot(xnx)
            plt.stem(pIndx.flatten(), [.01]*len(pIndx.flatten()) ,'r')
            plt.stem(nIndx.flatten(), [-.01]*len(nIndx.flatten()))
            maximum = np.max(xpx)
            total_nr_bits = 0
            for i in pIndx:
                nr_bit = (i[1]-i[0] + 1)//bitSamplePeriod
                plt.text(np.average(i)-20,maximum/5, str(nr_bit))
                total_nr_bits += nr_bit
            for i in nIndx:
                nr_bit = (i[1]-i[0] + 1)//bitSamplePeriod
                plt.text(np.average(i)-20,-maximum/5, str(nr_bit) )
                # total_nr_bits += nr_bit
            # print(total_nr_bits)
            plt.xlabel("sample")
            if title is not None:
                plt.title(title)
            plt.show()

        return res, resInHex, bitLen[0:-1], max_gradient_unwrapped_phase[0:-1]



class Utills: 

    # %%
    def readFile(self, file, Fc = None, Fs = None):
        IQsamples = np.fromfile(file, np.complex64)
        tindx = np.fromfile(file+".tindx",sep= ',')
        try:
            temp  = file.split('/')[-1] # handles the directory
            name, Fc_from_name, Fs_from_name, gain, acq_time, inchamber, extenstion = temp.split('_')
            if Fc is None:
                Fc = Fc_from_name
            if Fs is None:
                Fs = Fs_from_name
            IQdatas = IQdata(path = file, samples = IQsamples,tindx = tindx,Fc=int(float(Fc)),Fs=int(float(Fs)))
            print("File name has a correct format!")
        except:
            Fc = 2.444e9
            Fs = 100e6
            IQdatas = IQdata(path = file, samples = IQsamples,tindx = tindx,Fc=Fc ,Fs=Fs)
        return IQdatas,tindx


    def frameFinder(self, samples, minFrameSize = 1000):
        test_list = np.nonzero(samples)
        framesIndex = []
        for k, g in groupby(enumerate(test_list[0]), lambda ix: ix[0]-ix[1]):
            temp = list(map(itemgetter(1), g))
            if len(temp)< minFrameSize:
                continue
            framesIndex.append([temp[0],temp[-1]])
        return np.array(framesIndex)

    def zeroRemover(self, file,samples, framesIndex):
        f = open(file,"wb")
        f_time_index= open(file+".tindx","wb")
        framesIndex.tofile(f_time_index,sep= ',')
        for i,j in framesIndex:
            frame = samples[i:j]
            #if len(samples)< thresh:
                # continue
            frame.tofile(f)
            np.zeros(2,dtype=np.complex64).tofile(f)

        f.close()
        f_time_index.close()

    def frame_subtractor(self,x, y, offSet: int = 0):
            return x[offSet: min(len(x), len(y))] - y[0: min(len(x),len(y)) - offSet]

    def plotter(self,IQdata,tindx, batch, frameShowLimit = -1,info = True,fft = False,compression_ratio = 15, title = None):
        if frameShowLimit == -1 or frameShowLimit > len(IQdata.TotalFramesIndex):
            frameShowLimit = len(IQdata.TotalFramesIndex)
        if batch == -1:
            batch = frameShowLimit
        tindx_temp = []
        for i in range(len(tindx)//2):
            try:
                tindx_temp.append(tindx[2*i+2]-tindx[2*i+1])
            except:
                break
        start = 0 
        frameCnt = 0
        
        while frameCnt < frameShowLimit:
            x = []
            plt.figure(figsize=(20, 3), dpi=100)
            for i in range(batch):
                try:
                    frame  = IQdata.frameByNumber(frameCnt)
                except:
                    continue  
                
                # adding zeros based on the tindx file
                try:
                    strt_indx = len(x)
                    end_inx = len(x) + len(frame)
                    frameindx = [strt_indx,end_inx]
                    frame =  np.concatenate((frame,np.zeros(int(tindx_temp[frameCnt]//compression_ratio))))
                    if info:
                        chnl  = IQdata.channelDetection(frameCnt)
                        plt.text(np.average(frameindx)-len(x)//((frameCnt+1)*20),max(abs(frame))/5, "#"+str(frameCnt) )
                        plt.text(np.average(frameindx)-len(x)//(20*(frameCnt+1)),max(abs(frame))/20, "c:"+str(chnl) )
                        plt.stem(frameindx, [.02]*len(frameindx) ,'r')
                except:
                    frameCnt += 1
                    continue
                x = np.concatenate((x,frame))
                frameCnt += 1
            plt.plot(np.abs(x))
            plt.xlabel("time (s)")
            plt.ylabel("amplitude")
            if title is not None:
                plt.title(title)
            plt.show()
            plt.close()

            fftcnt = frameCnt - batch
            if fft:
                for i in range(batch):
                    plt.figure(figsize=(20, 3), dpi=100)
                    absfft= np.abs(IQdata.fft(fftcnt))
                    plt.plot(np.arange(-IQdata.Fs/2,IQdata.Fs/2,IQdata.Fs/len(absfft)) + IQdata.Fc,absfft)
                    plt.xlabel("Freq")
                    plt.ylabel("amplitude")
                    plt.title("frame" + str(fftcnt))
                    plt.show()
                    plt.close()
                    fftcnt +=1


            

# # test for the raw sample processing
# import src.dataProcessing as dataproccessing
# utills = dataproccessing.Utills()                
# chnl2 = "/home/moh/Documents/PhD/old/BLE_Dataset_Tool/DataSet/SDR_1/onBody/iter_1/x310-jan-14-HighPower-dvc-2-pos-moving_2440000000.0_100000000.0_31.5_2_1_1.iq"
# IQdatas1,tindx1 = utills.readFile(chnl2,Fc=2440000000.0 - 185000)

# utills.plotter(IQdata=IQdatas1,tindx=tindx1,batch=10,frameShowLimit=10,compression_ratio=15)
# IQdatas1.getMetaData(3)
        
        






