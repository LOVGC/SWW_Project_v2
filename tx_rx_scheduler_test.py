# plot and data structure packages
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from utils.signals import complex_sinusoid, generate_chirp

# threading package
import threading

# hardware API packages
import uhd
from uhd import libpyuhd as lib

# user defined application packages
from utils.Usrp_B210 import Usrp_B210


master_clock_rate = 10e6
samp_rate_ADC_DAC = 10e6
my_B210 = Usrp_B210(samp_rate_ADC_DAC, master_clock_rate)

# generate chirp data

#Bi = 1e6  # BW of the chirp
#Tp = 40e-3  # duration of the chirp
#tx_data = generate_chirp(samp_rate_ADC_DAC, Bi, Tp)

# generate sin-wave data
tx_data, length_wave_one_period = complex_sinusoid(samp_rate_ADC_DAC)

# test tx_waveform, ref: streaming Model code examples


center_freq = 1e9
tx_gains = [65, 80]



# prepare rx_buffer
num_rx_samps = tx_data.shape[1] * 10
# num_rx_samps = length_wave_one_period*20
rx_result = np.zeros((2, num_rx_samps), dtype=np.complex64)

# specify rx parameters
center_freq = 1e9
rx_gains = [20, 20]
rx_md = lib.types.rx_metadata()
# next we can try to send and receive the chirp and uisng an event for communicating between the two threads


tx_thread = threading.Thread(
    target=my_B210.tx_waveform, args=(tx_data, center_freq, tx_gains)
)
rx_thread = threading.Thread(
    target=my_B210.rx_waveform, args=(rx_result, center_freq, rx_gains)
)
rx_thread.start()
tx_thread.start()

tx_thread.join()
rx_thread.join()



print(f"num_rx_samps = {num_rx_samps}")
print(f"num_tx_samps = {tx_data.shape}")
# convert numpy data to matlab data
file_path = f"./data/matlab_data/sww_radar_data.mat"
with open(file_path, "wb") as f:  # need 'wb' in Python3
    scipy.io.savemat(f, {"sww_radar_data": rx_result})
    scipy.io.savemat(f, {"rx_gains": rx_gains})
    scipy.io.savemat(f, {"tx_data": tx_data})
