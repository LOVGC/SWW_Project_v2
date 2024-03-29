import uhd
from uhd import libpyuhd as lib

import numpy as np

import threading
import time
import sys

from utils.signals import complex_sinusoid, generate_chirp

import matplotlib.pyplot as plt

# define constants
INIT_DELAY = 0.05  # 50mS initial delay before transmit
TXRX_RATE = 10e6  # the ADC/DAC rate of the rx and tx

###############################################################################################
# define device parameters

# tx, rx channels
rx_channels = (0, 1)
tx_channels = (0, 1)

# tx, rx cpu and otw
tx_cpu = rx_cpu = "fc32"
tx_otw = rx_otw = "sc16"

# subdevice specification: https://files.ettus.com/manual/page_configuration.html
rx_subdev = tx_subdev = "A:A A:B"

duration = 2

#############################################################################################
# define the sww_tx_subpulse and sww_rx_subpulse
sww_tx_subpulse, _ = complex_sinusoid(TXRX_RATE)
num_rx_samps = sww_tx_subpulse.shape[1] * 15
# num_rx_samps = length_wave_one_period*20
sww_rx_subpulse = np.zeros((2, num_rx_samps), dtype=np.complex64)


##########################################################################################


def set_txrx_center_freq(usrp, target_center_freq):

    usrp.set_rx_freq(lib.types.tune_request(target_center_freq), 0)

    # wait until the lo's are locked, or maybe just put some time delay here?
    # while not (usrp.get_rx_sensor("lo_locked", 0).to_bool()):
    #     pass

    usrp.set_tx_freq(lib.types.tune_request(target_center_freq), 0)

    # wait until the lo's are locked, or maybe just put some time delay here?
    # while not (usrp.get_tx_sensor("lo_locked", 0).to_bool()):
    #     pass


def set_txrx_gains(usrp, tx_gains, rx_gains):
    usrp.set_tx_gain(tx_gains[0], 0)
    usrp.set_tx_gain(tx_gains[1], 1)

    usrp.set_rx_gain(rx_gains[0], 0)
    usrp.set_rx_gain(rx_gains[1], 1)


def tx_worker(usrp, tx_streamer, tx_stop_event, sww_tx_start_event, sww_rx_on_event):

    # Make a transmit buffer
    num_channels = tx_streamer.get_num_channels()
    max_samps_per_packet = tx_streamer.get_max_num_samps()

    transmit_buffer = np.zeros((num_channels, max_samps_per_packet), dtype=np.complex64)
    metadata = uhd.types.TXMetadata()
    metadata.time_spec = uhd.types.TimeSpec(
        usrp.get_time_now().get_real_secs() + INIT_DELAY
    )
    metadata.has_time_spec = True

    while not tx_stop_event.is_set():
        try:
            if sww_tx_start_event.is_set():
                send_samps = 0
                total_samps = sww_tx_subpulse.shape[1]
                metadata.has_time_spec = False

                # waiting for the rx to turn on
                sww_rx_on_event.wait()

                while send_samps < total_samps:
                    real_samps = min(max_samps_per_packet, total_samps - send_samps)
                    send_samps += tx_streamer.send(
                        sww_tx_subpulse[:, send_samps : (send_samps + real_samps)],
                        metadata,
                    )
                
                # sww_tx_start_event.clear()

            else:
                tx_streamer.send(transmit_buffer, metadata)
                metadata.has_time_spec = False

        except RuntimeError as ex:
            print(f"Runtime error in transmit: {ex}")

        # code for recovery from errors: ref benchmark_tx_rate_async_helper()

    # Send a mini EOB packet
    tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), metadata)


def rx_worker(usrp, rx_streamer, rx_stop_event, sww_rx_start_event, sww_rx_on_event):

    # make a receive buffer
    num_channels = rx_streamer.get_num_channels()
    max_samps_per_packet = rx_streamer.get_max_num_samps()

    recv_buffer = np.empty((num_channels, max_samps_per_packet), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    # craft and send the stream command
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = False
    stream_cmd.time_spec = uhd.types.TimeSpec(
        usrp.get_time_now().get_real_secs() + INIT_DELAY
    )
    rx_streamer.issue_stream_cmd(stream_cmd)

    while not rx_stop_event.is_set():
        try:
            if sww_rx_start_event.is_set():
                recv_samps = 0
                total_samps = sww_rx_subpulse.shape[1]
                while recv_samps < total_samps:
                    samps = rx_streamer.recv(
                        recv_buffer,
                        metadata,
                    )

                    sww_rx_on_event.set()

                    if samps:
                        real_samps = min(samps, total_samps - recv_samps)
                        sww_rx_subpulse[
                            :, recv_samps : recv_samps + real_samps
                        ] = recv_buffer[:, 0:real_samps]
                        recv_samps += real_samps
                    else:
                        print(metadata)
                sww_rx_on_event.clear()

            else:
                rx_streamer.recv(recv_buffer, metadata) * num_channels

        except RuntimeError as ex:
            print(f"Runtime Error in receive: {ex}")

        # handle receiver errors
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(metadata)

    # After we get the signal to stop, issue a stop command
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))


def sww_scheduler():
    pass


def main():
    # create device
    usrp = uhd.usrp.MultiUSRP("num_recv_frames=1000")

    # always select the subdevice first, the channel mapping affects the other settings
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(rx_subdev))
    usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(tx_subdev))

    # setup ref clock
    usrp.set_clock_source("internal")

    # setup pps
    usrp.set_time_source("internal")

    # we need to synchronize the channels by running the following
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

    # get the rx streamer
    usrp.set_rx_rate(TXRX_RATE)
    st_args = uhd.usrp.StreamArgs(rx_cpu, rx_otw)
    st_args.channels = rx_channels
    rx_streamer = usrp.get_rx_stream(st_args)

    # get the tx streamer
    usrp.set_tx_rate(TXRX_RATE)
    st_args = uhd.usrp.StreamArgs(tx_cpu, tx_otw)
    st_args.channels = tx_channels
    tx_streamer = usrp.get_tx_stream(st_args)

    # create threads
    threads = []

    rx_stop_event = threading.Event()
    sww_rx_start_event = threading.Event()
    sww_rx_on_event = threading.Event()

    rx_thread = threading.Thread(
        target=rx_worker,
        args=(usrp, rx_streamer, rx_stop_event, sww_rx_start_event, sww_rx_on_event),
    )
    rx_thread.setName("rx_thread")
    threads.append(rx_thread)
    ##############################################################################
    tx_stop_event = threading.Event()
    sww_tx_start_event = threading.Event()

    tx_thread = threading.Thread(
        target=tx_worker,
        args=(usrp, tx_streamer, tx_stop_event, sww_tx_start_event, sww_rx_on_event),
    )
    tx_thread.setName("tx_thread")
    threads.append(tx_thread)

    # start the threads
    # set center freqs and gains

    tx_gains = [75, 75]
    rx_gains = [10, 10]
    center_freq = 1e9

    set_txrx_gains(usrp, tx_gains, rx_gains)
    set_txrx_center_freq(usrp, center_freq)

    rx_thread.start()
    tx_thread.start()
    
    # wait for some time to make sure the tx_worker and rx_worker are working properly
    time.sleep(10 * INIT_DELAY)

    
    

    # start collect data
    sww_rx_start_event.set()
    sww_tx_start_event.set()

    time.sleep(duration)

    # turn off device
    print("Sending signal to stop!")
    rx_stop_event.set()
    tx_stop_event.set()
    for thr in threads:
        thr.join()

    plt.plot(np.real(sww_rx_subpulse[0, :]))
    plt.show()


if __name__ == "__main__":

    sys.exit(not main())
