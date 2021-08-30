# plot and data structure packages
from os import removexattr
from pickle import load
from numpy.core.defchararray import center
import scipy.io
import numpy as np
from utils.signals import complex_sinusoid, generate_chirp

# user defined application packages
from utils.Usrp_B210 import *
from utils.save_load import *

import matplotlib.pyplot as plt


###################################################################
# device parameters and radar parameters
samp_rate_ADC_DAC = 5e6
master_clock_rate = None # clock rate should be larger, less than 10MHz will cause problem
my_B210 = Usrp_B210(samp_rate_ADC_DAC, master_clock_rate)



txA_gain = 75
txB_gain = 75

rxA_gain = 10
rxB_gain = 10

# radar parameters
start_freq = 500e6
stop_freq = 3e9
freq_step = 10e6

# center_freqs = np.arange(start_freq, stop_freq, freq_step)

center_freqs = np.random.randint(start_freq, stop_freq, size=50)

####################################################################
# generate the sensing plan nparray
tx_gains = [txA_gain, txB_gain]
rx_gains = [rxA_gain, rxB_gain]


sensing_plan_list = []
for f in center_freqs:
    # prepare the TX() object
    baseband_waveform, length_wave_one_period = complex_sinusoid(samp_rate_ADC_DAC)
    center_freq = f
    print(f)

    tx = TX(baseband_waveform, center_freq, tx_gains)
    # prepare the RX() object
    num_rx_samps = baseband_waveform.shape[1] * 50
    # num_rx_samps = length_wave_one_period*20
    result = np.zeros((2, num_rx_samps), dtype=np.complex64)
    rx = RX(result, center_freq, rx_gains)

    sensing_plan_list.append((tx, rx))

sensing_plan_nparray_sine_waves = np.array(sensing_plan_list, dtype=object)

print(f"shape is {sensing_plan_nparray_sine_waves.shape}")

# turn on the AGC?? not sure whether this is a good idea

print(
    "Memory size of numpy array in bytes:",
   sensing_plan_nparray_sine_waves.size * sensing_plan_nparray_sine_waves.itemsize,
)


# for i in range(200):
#     print(f"{i} th iteration")
#     my_B210.sww_sensing(sensing_plan_nparray_sine_waves)

start = time.time()
my_B210.sww_sensing(sensing_plan_nparray_sine_waves)
end = time.time()
print(f"total scan time = {end - start}")

# save data
my_B210.save_sww_data("loopback_test")


