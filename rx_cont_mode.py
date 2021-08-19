import threading
import uhd
from uhd import libpyuhd as lib
import numpy as np
from threading import Thread
import time
import matplotlib.pyplot as plt

SAMP_RATE = 10e6  # sample rate of the ADC/DAC, for now need to >= the BW of the complex chirp

MASTER_CLOCK_RATE = 20e6

RX_DELAY = 0.01
TX_DELAY = 0.012

def setup_device(samp_rate, master_clock_rate):

    # create hte usrp device object
    args = "type = b200"
    usrp = uhd.usrp.MultiUSRP(args)

    # set clock ant time
    freq_clock_source = "internal"
    # this sets the source of the frequency reference, typically a 10 MHz signal
    usrp.set_clock_source(freq_clock_source)
    # this set the master clock rate
    usrp.set_master_clock_rate(master_clock_rate)

    # select subdevices: the RF frontend tx chains and rx chains
    subdevice = (
        "A:A A:B"  # select subdevice in the daughterboard, we are using two channels
    )
    subdevice_spec = lib.usrp.subdev_spec(subdevice)
    usrp.set_rx_subdev_spec(subdevice_spec)
    usrp.set_tx_subdev_spec(subdevice_spec)
    print(f"Using Device: {usrp.get_pp_string()}")

    # set sample rate of ADC/DAC
    channel_list = (0, 1)  # 0 represents channel A, 1 represents channel B
    # this will set over all channels
    usrp.set_tx_rate(samp_rate)
    usrp.set_rx_rate(samp_rate)
    print(f"Actual RX0 rate: {usrp.get_rx_rate(0) / 1e6} Msps")
    print(f"Actual RX1 rate: {usrp.get_rx_rate(1) / 1e6} Msps")
    print(f"Actual TX0 rate: {usrp.get_tx_rate(0) / 1e6} Msps")
    print(f"Actual TX1 rate: {usrp.get_tx_rate(1) / 1e6} Msps")

    # create stream args, tx streamer and rx streamer
    st_args = lib.usrp.stream_args("fc32", "sc16")  # do not use fc64!!!! will lead to segmentation default(core dumped!)
    st_args.channels = channel_list

    tx_streamer = usrp.get_tx_stream(st_args)  # create tx streamer
    rx_streamer = usrp.get_rx_stream(st_args)  # create rx streamer

    # init the usrp device time to zero
    usrp.set_time_now(lib.types.time_spec(0.0))

    return usrp, tx_streamer, rx_streamer

def set_tx_center_freq(usrp, target_center_freq):

    # tune center freqs on all channels, since the two tx ports share the same tx LO,
    # we only need to set one channel
    usrp.set_tx_freq(lib.types.tune_request(target_center_freq), 0)

    # wait until the lo's are locked, or maybe just put some time delay here?
    while not (usrp.get_tx_sensor("lo_locked", 0).to_bool()):
        pass


def set_rx_center_freq(usrp, target_center_freq):

    # set rx center freqs on all channels, since the two rx ports share the same rx LO,
    # we only need to set one channel
    usrp.set_rx_freq(lib.types.tune_request(target_center_freq), 0)

    # wait until the lo's are locked, or maybe just put some time delay here?
    while not (usrp.get_rx_sensor("lo_locked", 0).to_bool()):
        pass

tx_stop_event = threading.Event()
def tx_waveform(usrp, tx_streamer, tx_md, baseband_waveform, center_freq, tx_gains, tx_stop_event):

    # set the tx gains first
    usrp.set_tx_gain(tx_gains[0], 0)
    usrp.set_tx_gain(tx_gains[1], 1)

    # set the tx center freqs
    set_tx_center_freq(usrp, center_freq)

    # send the baseband_waveform
    while not tx_stop_event.is_set():
        tx_streamer.send(
            baseband_waveform, tx_md
        )  # sending the data takes some time, thus the program will be blocking here to run this




rx_stop_event = threading.Event()
def rx_waveform(usrp, rx_streamer, rx_md, output_result, center_freq, rx_gains, rx_stop_event):

    # set the rx gains first
    usrp.set_rx_gain(rx_gains[0], 0)
    usrp.set_rx_gain(rx_gains[1], 1)

    # set the rx center freqs
    set_rx_center_freq(usrp, center_freq)

    rx_buffer = np.zeros((2, 10*rx_streamer.get_max_num_samps()), dtype=np.complex64)
    # prepare the streamer
    stream_cmd = lib.types.stream_cmd(lib.types.stream_mode.start_cont)
    stream_cmd.stream_now = False
    stream_cmd.time_spec = usrp.get_time_now() + lib.types.time_spec(RX_DELAY)
    rx_streamer.issue_stream_cmd(stream_cmd)  # tells all channels to stream

    # fetch the data from rxA and rxB into the rx_buffer
    while not rx_stop_event.is_set():
        try:
            num_samps = rx_streamer.recv(rx_buffer, rx_md)
        except:
            print("Run time error")
        if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
            print(rx_md)
            print(num_samps)


def stop_rx():
    stream_cmd = lib.types.stream_cmd(lib.types.stream_mode.stop_cont)
    stream_cmd.num_samps = num_rx_samps
    stream_cmd.stream_now = False
    stream_cmd.time_spec = usrp.get_time_now() + lib.types.time_spec(RX_DELAY)
    rx_streamer.issue_stream_cmd(stream_cmd)

def generate_tx_data():

    # create a chirp signal
    Bi = 5e6  # BW of the chirp
    Tp = 4e-3  # duration of the chirp
    K = Bi / Tp  # chirp rate
    A = 0.75  # amplitude of the chirp

    samp_freq = Bi
    num_samps = samp_freq * Tp
    t = np.linspace(-Tp / 2, Tp / 2, int(num_samps))
    w_t = A * np.exp(1j * np.pi * K * t ** 2)

    # prepare a 2 by N data
    tx_data = np.tile(w_t, (2, 1))

    return tx_data


usrp, tx_streamer, rx_streamer = setup_device(SAMP_RATE, MASTER_CLOCK_RATE)

# prepare tx data, tx_md
baseband_waveform = generate_tx_data()



# prepare rx_buffer, rx_md
num_rx_samps = 15 * baseband_waveform[0].size
output_result = np.zeros((2, num_rx_samps), dtype=np.complex64)

rx_md = lib.types.rx_metadata()


# define tx device parameters
tx_center_freq = 1e9
tx_gains = [50, 50]

# define rx device parameters
rx_center_freq = 1e9
rx_gains = [20, 20]
tx_md = lib.types.tx_metadata()
tx_md.start_of_burst = True
tx_md.end_of_burst = False
tx_md.has_time_spec = False
# tx_md.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + TX_DELAY)

tx_worker = threading.Thread(
    target=tx_waveform,
    args=(usrp, tx_streamer, tx_md, baseband_waveform, tx_center_freq, tx_gains, tx_stop_event),
)

rx_worker = threading.Thread(target=rx_waveform, args=(usrp, rx_streamer, rx_md, output_result, rx_center_freq, rx_gains, rx_stop_event))

tx_worker.start()
rx_worker.start()
