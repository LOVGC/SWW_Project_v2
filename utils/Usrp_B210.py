import threading
from numpy.lib import utils
import uhd
from uhd import libpyuhd as lib
import numpy as np
from threading import Thread
import time

import requests
import matplotlib.pyplot as plt

class Usrp_B210:

    ###################################################################################################################
    # Define some staticmethod in the class as utility functions
    ###################################################################################################################

    #################################################################################################################
    # Initialization for SWW Radar implemented with USRP B210
    ################################################################################################################
    def __init__(self, samp_rate_ADC_DAC, master_clock_rate=30e6):
        """This function implements the common settings shared by all the subpulses. The common settings are
            Maximum BW for all Baseband subpulses

        Args:
            samp_rate_ADC_DAC (float): the sample rate of the ADC/DAC,
                                       suppose your baseband signal is complex valued
                                       and has bandwidth(spectrum support) of B, then your samp_rate_ADC_DAC
                                       should be at least B.
            master_clock_rate (float): the base clock rate that is used as a reference clock for the ADC/DAC
                                       and the FPGA, should be at least samp_rate_ADC_DAC.
        """
        # create the usrp object
        # makes the receive buffer much larger (the default value is 32), helps to reduce overflows
        # args = "type=b200"
        # self.usrp = uhd.usrp.MultiUSRP(args)

        self.usrp = uhd.usrp.MultiUSRP(
            "num_recv_frames=1000"
        )  # increase this to avoid rx overflow
        # set clock ant time
        freq_clock_source = "internal"
        # this sets the source of the frequency reference, typically a 10 MHz signal
        self.usrp.set_clock_source(freq_clock_source)
        # this set the master clock rate
        self.usrp.set_master_clock_rate(master_clock_rate)

        # select subdevices: the RF frontend tx chains and rx chains
        subdevice = "A:A A:B"  # select subdevice in the daughterboard, we are using two channels
        subdevice_spec = lib.usrp.subdev_spec(subdevice)
        self.usrp.set_rx_subdev_spec(subdevice_spec)
        self.usrp.set_tx_subdev_spec(subdevice_spec)
        print(f"Using Device: {self.usrp.get_pp_string()}")

        # set sample rate of ADC/DAC
        channel_list = (0, 1)  # 0 represents channel A, 1 represents channel B
        # this will set over all channels
        self.usrp.set_tx_rate(samp_rate_ADC_DAC)
        self.usrp.set_rx_rate(samp_rate_ADC_DAC)
        print(f"Actual RX0 rate: {self.usrp.get_rx_rate(0) / 1e6} Msps")
        print(f"Actual RX1 rate: {self.usrp.get_rx_rate(1) / 1e6} Msps")
        print(f"Actual TX0 rate: {self.usrp.get_tx_rate(0) / 1e6} Msps")
        print(f"Actual TX1 rate: {self.usrp.get_tx_rate(1) / 1e6} Msps")

        # set the filter

        # create stream args, tx streamer and rx streamer
        st_args = lib.usrp.stream_args("fc32", "sc16")
        st_args.channels = channel_list

        self.tx_streamer = self.usrp.get_tx_stream(st_args)  # create tx streamer
        self.rx_streamer = self.usrp.get_rx_stream(st_args)  # create rx streamer

        # the delay in unit of seconds for turn on tx_streamer and rx_streamer
        self.RX_DELAY = 0.01
        self.TX_DELAY = 0.012

        # create events for better sync
        self.rx_on_event = threading.Event()
        self.tx_finish_event = threading.Event()

        # init the usrp device time to zero
        self.usrp.set_time_now(lib.types.time_spec(0.0))

    ##################################################################################

    ##################################################################################
    # Implement Some useful functions for control the USRP Device
    ##################################################################################

    def set_tx_center_freq(self, target_center_freq):
        """this function tune the tx LO to the target_center_freq
            wait until the tx LO is locked.

        Args:
            target_center_freq (float): 70MHz ~ 6GHz
        """

        # tune center freqs on all channels, since the two tx ports share the same tx LO,
        # we only need to set one channel
        self.usrp.set_tx_freq(lib.types.tune_request(target_center_freq), 0)

        # wait until the lo's are locked, or maybe just put some time delay here?
        while not (self.usrp.get_tx_sensor("lo_locked", 0).to_bool()):
            pass

    def set_rx_center_freq(self, target_center_freq):
        """this function tune the tx LO to the target_center_freq
            wait until the rx LO is locked.

        Args:
            target_center_freq (float): 70MHz ~ 6GHz
        """

        # set rx center freqs on all channels, since the two rx ports share the same rx LO,
        # we only need to set one channel
        self.usrp.set_rx_freq(lib.types.tune_request(target_center_freq), 0)

        # wait until the lo's are locked, or maybe just put some time delay here?
        while not (self.usrp.get_rx_sensor("lo_locked", 0).to_bool()):
            pass

    ###################################################################################

    ##################################################################################
    # the main transmitter function and receiver function
    ##################################################################################

    def tx_waveform(self, baseband_waveform, center_freq, tx_gains):
        """this function send the baseband_waveform to the two tx channels, at the specified
            center_freq and using the tx_gains

        Args:
            baseband_waveform (np complex64 array with shape 2 by N):
                1) the elements are of type complex64 and
                2) the first row of I/Q data is sent to txA
                   and the second row of I/Q data is sent to txB
            center_freq (float): 70MHz ~ 6GHz

            tx_gains (python list of floats): [<tx_gain for txA>, <tx_gain for txB>],
                                              Gain range PGA: 0.0 to 89.8 step 0.2 dB
        """

        self.usrp.set_tx_gain(tx_gains[0], 0)
        self.usrp.set_tx_gain(tx_gains[1], 1)

        # set the tx center freqs
        self.set_tx_center_freq(center_freq)

        # prepare tx metadata
        # for tx, we want to transmit a signal at some time and done
        tx_md = lib.types.tx_metadata()
        tx_md.has_time_spec = True

        self.rx_on_event.wait()

        self.tx_finish_event.clear()

        tx_md.time_spec = uhd.types.TimeSpec(
            self.usrp.get_time_now().get_real_secs() + self.TX_DELAY
        )

        # send the baseband_waveform
        self.tx_streamer.send(
            baseband_waveform, tx_md
        )  # sending the data takes some time, thus the program will take some time to run this

        # tell the device to stop transmitting
        tx_md.end_of_burst = True
        self.tx_streamer.send(np.zeros((2, 0), dtype=np.complex64), tx_md)

        self.tx_finish_event.set()

    def rx_waveform(self, result, center_freq, rx_gains):
        """[summary] fetch the data from rxA and rxB into the result at center_freq and using
                    the specified rx_gains.

        Args:
            result ([type]: 2 by N numpy array of dtype = np.complex64): used to store the data
                            from rxA in row 0, and rxB in row 1.
            center_freq ([type]: float): 70MHz ~ 6GHz
            rx_gains ([type]: python list of floats): [<rx_gain for rxA>, <rx_gain for rxB>],
                                              Gain range PGA: 0.0 to 76.0 step 1.0 dB
        """
        # set the rx gains
        self.usrp.set_rx_gain(rx_gains[0], 0)
        self.usrp.set_rx_gain(rx_gains[1], 1)
        # set the rx center freqs
        self.set_rx_center_freq(center_freq)

        rx_md = lib.types.rx_metadata()

        # prepare the streamer
        stream_cmd = lib.types.stream_cmd(lib.types.stream_mode.num_done)
        num_rx_samps = result.shape[1]
        stream_cmd.num_samps = num_rx_samps
        stream_cmd.stream_now = False
        stream_cmd.time_spec = uhd.types.TimeSpec(
            self.usrp.get_time_now().get_real_secs() + self.RX_DELAY
        )
        self.rx_streamer.issue_stream_cmd(stream_cmd)  # tells all channels to stream

        self.rx_on_event.set()
        # fetch the data from rxA and rxB into the result
        self.rx_streamer.recv(result, rx_md)

        self.tx_finish_event.wait()
        self.rx_on_event.clear()

        # turn off the rx_streamer
        stream_cmd = lib.types.stream_cmd(lib.types.stream_mode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stream_cmd)

    def tx_rx_one_cycle(self, tx, rx):

        tx_thread = threading.Thread(
            target=self.tx_waveform,
            args=(tx.baseband_waveform, tx.center_freq, tx.tx_gains),
        )

        rx_thread = threading.Thread(
            target=self.rx_waveform, args=(rx.result, rx.center_freq, rx.rx_gains)
        )
        rx_thread.start()
        tx_thread.start()

        tx_thread.join()
        rx_thread.join()
        print(f"tx-rx done for center freq = {tx.center_freq}")

    def sww_sensing(self, sensing_plan_nparray):
        """[summary]

        Args:
            sensing_plan_nparray ([type] numpy object array): created like this:
            np.array([(tx1, rx1), (tx2, rx2), (tx3, rx3)],dtype=object)
        """
        for sensing_plan in sensing_plan_nparray:
            start = time.time()

            tx = sensing_plan[0]
            rx = sensing_plan[1]
            self.tx_rx_one_cycle(tx, rx)

            end = time.time()
            print(f"one subpulse takes = {end - start}")

            plt.plot(
                
                np.real(rx.result[0, :]),
            )
            plt.show()


class TX:
    def __init__(self, baseband_waveform, center_freq, tx_gains):
        self.baseband_waveform = baseband_waveform
        self.center_freq = center_freq
        self.tx_gains = tx_gains


class RX:
    def __init__(self, result, center_freq, rx_gains) -> None:
        self.result = result
        self.center_freq = center_freq
        self.rx_gains = rx_gains


# usage example
if __name__ == "__main__":

    # construct a MySWW object
    samp_rate_ADC_DAC = 30e6
    master_clock_rate = 30e6
    sww_radar = Usrp_B210(samp_rate_ADC_DAC, master_clock_rate)

    # test set center freqs
    sww_radar.set_rx_center_freq(1.5e9)
    print(f"rx center freq is {sww_radar.usrp.get_rx_freq(0)}")
    sww_radar.set_tx_center_freq(2.5e9)
    print(f"tx center freq is {sww_radar.usrp.get_tx_freq(0)}")

    # the bandwidth of an anolog lowpass filter can be thought of as its cutoff frequency
    print(
        f"Default tx LP filter bandwidth at txA is {sww_radar.usrp.get_tx_bandwidth(0)}"
    )  # the default baseband lowpass Filter's bandwidth is 56MHz
    print(
        f"Default tx LP filter bandwidth at txB is {sww_radar.usrp.get_tx_bandwidth(1)}"
    )

    print(
        f"Default rx LP filter bandwidth at rxA is {sww_radar.usrp.get_rx_bandwidth(0)}"
    )
    print(
        f"Default rx LP filter bandwidth at rxA is {sww_radar.usrp.get_rx_bandwidth(1)}"
    )

    # get the rx streamer buffer size
    print(f"the max size of rx buffer is {sww_radar.rx_streamer.get_max_num_samps()}")

    # turn on rx_streamer
    print("turn on the rx_streamer")
    sww_radar.turn_on_rx_streamer(1e9, [10, 10])

    print("turn off the rx_streamer")
    sww_radar.turn_off_rx_streamer()

    # test tx_waveform()

    tx_data = 0.5 * np.random.random((2, 1000))
    sww_radar.tx_waveform(tx_data, 2e9, [50, 40])
    print("sending data")

    # test rx_waveform()
    rx_buffer = np.zeros((2, 1000), dtype=np.complex64)

    sww_radar.rx_waveform(rx_buffer, 1e9, [50, 50])
    print(rx_buffer[:, 0:5])
