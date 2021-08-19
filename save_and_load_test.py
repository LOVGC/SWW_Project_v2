# plot and data structure packages
import scipy.io
import numpy as np
from utils.signals import complex_sinusoid, generate_chirp

# user defined application packages
from utils.Usrp_B210 import *
from utils.save_load import *

import matplotlib as plt

samp_rate_ADC_DAC = 15e6
# test save and load data with numpy array of object
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


sensing_plan_nparray = np.array([(tx1, rx1), (tx2, rx2)],dtype=object)

save_sensing_plan_nparray('test_save', sensing_plan_nparray)

sensing_plan_nparray2 = load_sensing_plan_nparray('test_save')
print(sensing_plan_nparray2)