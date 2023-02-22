import uhd, os, time
import numpy as np
import sys

mimo = True
samps_per_packet = 1950
# file_name = input("enter the file name")
# file_name = "FDATA/SDR_2/Moving/x310-jan-15-server-pos-5-client-1-pos-6-initialization-highpower"
file_name = "FDATA/SDR_1/Moving/x310-jan-14-HighPower-dvc-"+sys.argv[1]+ "-pos-"+sys.argv[2]
# file_name = "FDATA/SDR_1/Moving/x310-jan-14-noiseFloor3"
acq_time = 2# in second
inChamber = 1

Fc = 2.440e9# Hz
Fs = 100e6 # 200e6 does not
gain = 31.5 # dBs

THRESHOLD = 0.01
# THRESHOLD = 0
minSize = 30e6
maxSize = 99.99e6
minFrames = 100
maxFrames = 1000

if mimo: 
    chnls = [0,1]
else:
    chnls = [0]
args = "type = x300, addr = 192.168.30.2, second_addr = 192.168.40.2 "
usrp = uhd.usrp.MultiUSRP(args=args)
# uhd.usrp.SubdevSpec("0:A  1:D")



# this might change later due to wrong Fc or Fs selection and actual value will be replaced
file = file_name + "_"+str(Fc)+"_"+str(Fs)+"_"+str(gain)+"_"+str(acq_time) + "_"+ str(inChamber)+"_.iq"

def _config_streamer(usrp,chnls,spp = 200):
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = chnls
    st_args.args = "spp="+str(spp)
    streamer = usrp.get_rx_stream(st_args)

    return streamer



    rx_rate(args.rx_rate)
    st_args = uhd.usrp.StreamArgs(args.rx_cpu, args.rx_otw)
    st_args.channels = rx_channels
    rx_streamer = usrp.get_rx_stream(st_args)

def _batch_init(streamer,batch_size = None):
    if batch_size is None:
        batch_size = streamer.get_max_num_samps()
    nr_batches= int(acq_time * Fs / batch_size)
    return batch_size, nr_batches

def _start_stream(streamer,batch_size):
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.num_samps = batch_size
    stream_cmd.stream_now = False  
    stream_cmd.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + 0.05)   
    streamer.issue_stream_cmd(stream_cmd)

def _stop_stream(streamer,recv_buffer):
    metadata = uhd.types.RXMetadata()
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)
    while streamer.recv(recv_buffer, metadata):
        pass



# %%


usrp.set_time_now(uhd.types.TimeSpec(0.0)) # this should work well for syncing the MIMO channel

streamer = _config_streamer(usrp=usrp, chnls=chnls,spp=samps_per_packet)
batch_size, nr_batches = _batch_init(streamer=  streamer, batch_size= samps_per_packet)
print(batch_size)
recv_buffer = np.zeros((len(chnls), batch_size), dtype=np.complex64)
metadata = uhd.types.RXMetadata()

for chnl in chnls:
    usrp.set_rx_rate(Fs, chnl)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(Fc), chnl)
    usrp.set_rx_gain(gain, chnl)

# usrp.set_rx_agc(False, 0)

_start_stream(streamer = streamer,batch_size= batch_size)

#updating file name if mimo
if mimo:
    file1 = file_name + "_"+str(usrp.get_rx_freq())+"_"+str(usrp.get_rx_rate())+"_"+str(gain)+"_"+str(acq_time) + "_"+ str(inChamber)+"_1.iq"
    file2 = file_name + "_"+str(usrp.get_rx_freq())+"_"+str(usrp.get_rx_rate())+"_"+str(gain)+"_"+str(acq_time) + "_"+ str(inChamber)+"_2.iq"
    f1 = open(file1,"wb")
    f2 = open(file2,"wb")

    start = time.time()
    for i in range(nr_batches):
        streamer.recv(recv_buffer, metadata)
        # np.zeros(1).tofile(f)
        recv_buffer[0].tofile(f1)
        recv_buffer[1].tofile(f2)
    duration = time.time() - start
    print("\n Recorded Time: " + str(duration-0.05))
    # Stop Stream
    _stop_stream(streamer=streamer, recv_buffer=recv_buffer)
    f1.close()
    f2.close()

else:
    file = file_name + "_"+str(usrp.get_rx_freq())+"_"+str(usrp.get_rx_rate())+"_"+str(gain)+"_"+str(acq_time) + "_"+ str(inChamber)+"_.iq"
    f = open(file,"wb")

    start = time.time()
    for i in range(nr_batches):
        streamer.recv(recv_buffer, metadata)
        # np.zeros(1).tofile(f)
        recv_buffer[0].tofile(f)
    duration = time.time() - start
    print("\n Recorded Time: " + str(duration-0.05))
    # Stop Stream
    _stop_stream(streamer=streamer, recv_buffer=recv_buffer)
    f.close()



# %%
import dataProcessing 
utills = dataProcessing.Utills()

# %%


def file_santizer(file, minSize = 30e6,maxSize = 1e9,minFrames = 100,maxFrames = 1000,THRESHOLD = THRESHOLD):
    if THRESHOLD != 0:
        samples = np.fromfile(file, np.complex64) # Read in file.  We have to tell it what format it is
        zeros = np.abs(samples)<THRESHOLD
        samples[zeros] = 0
        framesIndex = utills.frameFinder(samples)
        utills.zeroRemover(file = file, samples=samples,framesIndex=framesIndex)     

        #check minimum number of the frames
        nr_frame = len(framesIndex)
        print("\nnumber of frames: " + str(nr_frame))
        if  nr_frame < minFrames or  nr_frame > maxFrames:
            print("# Frame check failed ...")
        else:
            print("# Frame OK ...")


    # Check the file size
    size = os.path.getsize(file)
    print("\nfile size is: " + str(os.path.getsize(file)))
    if size < minSize or size > maxSize:
        print("Size check failed ...    ")
    else:
        print("Size OK .")




### tests

if mimo:
    file_santizer(file=file1,minSize = 30e6,maxSize = 1e9,minFrames = 100,maxFrames = 1000)
    file_santizer(file=file2,minSize = 30e6,maxSize = 1e9,minFrames = 100,maxFrames = 1000)
else:
    file_santizer(file=file,minSize = 30e6,maxSize = 1e9,minFrames = 100,maxFrames = 1000)

print("Testing receive rate at Fc: {:.3f} GHz and Fs: {:.0f} Msps on {:d} channels.".format(
    usrp.get_rx_freq()/1e9, usrp.get_rx_rate()/1e6, streamer.get_num_channels()))
