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
from utils.Usrp_B210 import *

master_clock_rate = 15e6
samp_rate_ADC_DAC = 15e6
my_B210 = Usrp_B210(samp_rate_ADC_DAC, master_clock_rate)


tx_data, length_wave_one_period = complex_sinusoid(samp_rate_ADC_DAC)
center_freq = 1e9
tx_gains = [65, 80]


# prepare rx_buffer
num_rx_samps = tx_data.shape[1] * 10
# num_rx_samps = length_wave_one_period*20
rx_result = np.zeros((2, num_rx_samps), dtype=np.complex64)

# specify rx parameters
center_freq = 1e9
rx_gains = [20, 20]

tx1 = TX(tx_data, center_freq, tx_gains)
rx1 = RX(rx_result,center_freq, rx_gains)

#######################################################
tx_data2, length_wave_one_period2 = complex_sinusoid(samp_rate_ADC_DAC)
center_freq2 = 2e9
tx_gains2 = [70, 80]


# prepare rx_buffer
num_rx_samps = tx_data.shape[1] * 10
# num_rx_samps = length_wave_one_period*20
rx_result2 = np.zeros((2, num_rx_samps), dtype=np.complex64)

# specify rx parameters
center_freq2 = 2e9
rx_gains2 = [20, 20]


tx2 = TX(tx_data2, center_freq2, tx_gains2)
rx2 = RX(rx_result2,center_freq2, rx_gains2)



################################################
Bi = 5e6
Tp = 4e-3
tx_data3 = generate_chirp(samp_rate_ADC_DAC, Bi, Tp)
center_freq3 = 3e9
tx_gains3 = [70, 80]


# prepare rx_buffer
num_rx_samps = tx_data3.shape[1] * 10
# num_rx_samps = length_wave_one_period*20
rx_result3 = np.zeros((2, num_rx_samps), dtype=np.complex64)

# specify rx parameters

rx_gains3 = [20, 20]


tx3 = TX(tx_data3, center_freq3, tx_gains3)
rx3 = RX(rx_result3,center_freq3, rx_gains3)


sensing_plan_list = [(tx1, rx1), (tx2, rx2), (tx3, rx3)]
my_B210.sww_sensing(sensing_plan_list)

# convert numpy data to matlab data
file_path = f"./data/matlab_data/sww_radar_data1.mat"
with open(file_path, "wb") as f:  # need 'wb' in Python3
    scipy.io.savemat(f, {"sww_radar_data": rx1.result})
    scipy.io.savemat(f, {"rx_gains": rx1.rx_gains})
    scipy.io.savemat(f, {"tx_data": tx1.baseband_waveform})


file_path = f"./data/matlab_data/sww_radar_data2.mat"
with open(file_path, "wb") as f:  # need 'wb' in Python3
    scipy.io.savemat(f, {"sww_radar_data": rx2.result})
    scipy.io.savemat(f, {"rx_gains": rx2.rx_gains})
    scipy.io.savemat(f, {"tx_data": tx2.baseband_waveform})


file_path = f"./data/matlab_data/sww_radar_data3.mat"
with open(file_path, "wb") as f:  # need 'wb' in Python3
    scipy.io.savemat(f, {"sww_radar_data": rx3.result})
    scipy.io.savemat(f, {"rx_gains": rx3.rx_gains})
    scipy.io.savemat(f, {"tx_data": tx3.baseband_waveform})