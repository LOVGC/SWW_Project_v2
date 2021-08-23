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
TXRX_RATE = 1e6  # the ADC/DAC rate of the rx and tx

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
# Global data
sww_tx_subpulse, _ = complex_sinusoid(TXRX_RATE)
num_rx_samps = sww_tx_subpulse.shape[1] * 25
sww_rx_subpulse = np.zeros((2, num_rx_samps), dtype=np.complex64)

tx_gains = [75, 75]
rx_gains = [10, 10]
target_center_freq = 2e9


##########################################################################################
# useful data structure
class SensingPlan:
    def __init__(
        self,
        center_freq,
        tx_baseband_signal,
        tx_gains_list,
        rx_baseband_signal,
        rx_gains_list,
    ):
        self.center_freq = center_freq
        self.tx_baseband_signal = tx_baseband_signal
        self.tx_gains_list = tx_gains_list
        self.rx_baseband_signal = rx_baseband_signal
        self.rx_gains_list = rx_gains_list


###########################################################################################


def tx_worker(
    usrp,
    tx_streamer,
    tx_stop_event,
    sww_start_event,
    sww_rx_on_event,
    sww_tx_ready_event,
    sww_tx_done_event,
):

    # define state constant
    TX_ZEROS = 0
    TX_PREP = 1
    WAITING_TX_LO = 2
    SWW_TX_READY = 3
    SWW_TX_SUBPULSE = 4
    WAIT_FOR_SWW_RX_DONE = 5
    SWW_TX_DONE = 6
    TX_STOP = 7

    START_STATE = TX_ZEROS

    # initialize the state machine
    # Make a transmit buffer
    num_channels = tx_streamer.get_num_channels()
    max_samps_per_packet = tx_streamer.get_max_num_samps()

    transmit_buffer = np.zeros((num_channels, max_samps_per_packet), dtype=np.complex64)
    metadata = uhd.types.TXMetadata()
    metadata.time_spec = uhd.types.TimeSpec(
        usrp.get_time_now().get_real_secs() + INIT_DELAY
    )
    metadata.has_time_spec = True

    # start the state machine
    current_state = START_STATE
    while True:

        if current_state == TX_ZEROS:
            # actions:
            tx_streamer.send(transmit_buffer, metadata)
            metadata.has_time_spec = False

            # state transition
            if sww_start_event.is_set():
                current_state = TX_PREP

            if tx_stop_event.is_set():
                current_state = TX_STOP

        elif current_state == TX_PREP:
            # actions
            tx_streamer.send(transmit_buffer, metadata)

            usrp.set_tx_gain(tx_gains[0], 0)  # set tx gains
            usrp.set_tx_gain(tx_gains[1], 1)

            usrp.set_tx_freq(
                lib.types.tune_request(target_center_freq), 0
            )  # tune center freqs

            # transitions
            current_state = WAITING_TX_LO

        elif current_state == WAITING_TX_LO:
            # actions
            tx_streamer.send(transmit_buffer, metadata)

            # transitions
            if usrp.get_tx_sensor(
                "lo_locked", 0
            ).to_bool():  # if the tx lo locked, go to next state
                current_state = SWW_TX_READY

        elif current_state == SWW_TX_READY:
            # actions
            tx_streamer.send(transmit_buffer, metadata)

            sww_tx_ready_event.set()

            # transitions
            if sww_rx_on_event.is_set():
                current_state = SWW_TX_SUBPULSE
                sww_rx_on_event.clear()

        elif current_state == SWW_TX_SUBPULSE:
            send_samps = 0
            total_samps = sww_tx_subpulse.shape[1]

            while send_samps < total_samps:
                real_samps = min(max_samps_per_packet, total_samps - send_samps)
                send_samps += tx_streamer.send(
                    sww_tx_subpulse[:, send_samps : (send_samps + real_samps)],
                    metadata,
                )

            # state transitions
            current_state = WAIT_FOR_SWW_RX_DONE

        elif current_state == WAIT_FOR_SWW_RX_DONE:
            # actions
            tx_streamer.send(transmit_buffer, metadata)

            # state transitions
            if not sww_rx_on_event.is_set():
                current_state = SWW_TX_DONE

        elif current_state == SWW_TX_DONE:
            # actions
            sww_tx_done_event.set()
            sww_start_event.clear()

            tx_streamer.send(transmit_buffer, metadata)

            # transition
            current_state = TX_ZEROS

        elif current_state == TX_STOP:
            print("Stop TX")
            # send a mini EOB packet
            tx_streamer.send(np.zeros((num_channels, 0), dtype=np.complex64), metadata)
            break

        else:
            raise Exception(f"Unknown tx state: {current_state}")


def rx_worker(
    usrp,
    rx_streamer,
    rx_stop_event,
    sww_start_event,
    sww_rx_on_event,
    sww_tx_ready_event,
    sww_rx_done_event,
):

    # define state constants
    RX_ZEROS = 0
    RX_PREP = 1
    WAITING_RX_LO = 2
    SWW_RX_READY = 3
    SWW_RX_SUBPULSE = 4
    SWW_RX_DONE = 5
    RX_STOP = 6

    START_STATE = RX_ZEROS

    # state machine prepares
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

    # start the state machines
    current_state = START_STATE
    while True:

        if current_state == RX_ZEROS:
            # actions
            rx_streamer.recv(recv_buffer, metadata)

            # state transitions
            if sww_start_event.is_set():
                current_state = RX_PREP

            if rx_stop_event.is_set():
                current_state = RX_STOP

        elif current_state == RX_PREP:
            # actions
            rx_streamer.recv(recv_buffer, metadata)

            usrp.set_rx_gain(rx_gains[0], 0)  # set rx gains
            usrp.set_rx_gain(rx_gains[1], 1)

            usrp.set_rx_freq(lib.types.tune_request(target_center_freq), 0)

            # transitions
            current_state = WAITING_RX_LO

        elif current_state == WAITING_RX_LO:
            # actions
            rx_streamer.recv(recv_buffer, metadata)

            # transitions
            if usrp.get_rx_sensor("lo_locked", 0).to_bool():
                current_state = SWW_RX_READY

        elif current_state == SWW_RX_READY:
            # actions
            rx_streamer.recv(recv_buffer, metadata)

            # transitions
            if sww_tx_ready_event.is_set():
                current_state = SWW_RX_SUBPULSE
                sww_tx_ready_event.clear()

        elif current_state == SWW_RX_SUBPULSE:
            # actions
            recv_samps = 0
            total_samps = sww_rx_subpulse.shape[1]

            sww_rx_on_event.set()  # set this event on after receiving the first package
            while recv_samps < total_samps:
                samps = rx_streamer.recv(
                    recv_buffer,
                    metadata,
                )

                

                # save the rx signal
                if samps:
                    real_samps = min(samps, total_samps - recv_samps)
                    sww_rx_subpulse[
                        :, recv_samps : recv_samps + real_samps
                    ] = recv_buffer[:, 0:real_samps]
                    recv_samps += real_samps
                else:
                    print(metadata)

                if rx_stop_event.is_set():  # !!!!!!!!!!!!!!!!!!!!!do we need this?
                    break

            # transitions
            sww_rx_on_event.clear()
            current_state = SWW_RX_DONE

        elif current_state == SWW_RX_DONE:
            # actions
            sww_start_event.clear()
            sww_rx_done_event.set()
            rx_streamer.recv(recv_buffer, metadata)

            # transitions
            current_state = RX_ZEROS

        elif current_state == RX_STOP:
            print("Stop Rx")
            rx_streamer.issue_stream_cmd(
                uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            )
            break

        else:
            raise Exception(f"Unknown rx state: {current_state}")


def sww_data_collecter(usrp, tx_streamer, rx_streamer, sensing_plan_list):

    global sww_tx_subpulse, sww_rx_subpulse, tx_gains, rx_gains, target_center_freq
    # create threads and events
    threads = []

    sww_rx_on_event = threading.Event()
    sww_tx_ready_event = threading.Event()

    rx_stop_event = threading.Event()
    tx_stop_event = threading.Event()

    sww_start_event = threading.Event()
    sww_rx_done_event = threading.Event()
    sww_tx_done_event = threading.Event()

    rx_thread = threading.Thread(
        target=rx_worker,
        args=(
            usrp,
            rx_streamer,
            rx_stop_event,
            sww_start_event,
            sww_rx_on_event,
            sww_tx_ready_event,
            sww_rx_done_event,
        ),
    )
    rx_thread.setName("rx_thread")
    threads.append(rx_thread)
    ##############################################################################

    tx_thread = threading.Thread(
        target=tx_worker,
        args=(
            usrp,
            tx_streamer,
            tx_stop_event,
            sww_start_event,
            sww_rx_on_event,
            sww_tx_ready_event,
            sww_tx_done_event,
        ),
    )
    tx_thread.setName("tx_thread")
    threads.append(tx_thread)

    # start the threads
    rx_thread.start()
    time.sleep(
        5 * INIT_DELAY
    )  # wait for some time to make sure the tx_worker and rx_worker are working properly
    tx_thread.start()
    time.sleep(5 * INIT_DELAY)

    ##########################################################
    # after this line the tx_worker and rx_worker should all be in the START_STATE
    total_time_start = time.time()
    for sensing_plan in sensing_plan_list:
        # reload the global buffer
        sww_tx_subpulse = sensing_plan.tx_baseband_signal
        sww_rx_subpulse = sensing_plan.rx_baseband_signal
        tx_gains = sensing_plan.tx_gains_list
        rx_gains = sensing_plan.rx_gains_list
        target_center_freq = sensing_plan.center_freq
        print(f"working on center freq = {target_center_freq}")
        # reset the event
        sww_rx_done_event.clear()
        sww_tx_done_event.clear()

        start = time.time()
        # spawn the tx_worker and rx_worker
        sww_start_event.set()

        # wait for the tx_worker() and rx_worker() to complete their job
        
        sww_rx_done_event.wait()
        sww_tx_done_event.wait()

        end = time.time()
        print(f"takes time = {end - start}")

        # plot some stuff
        plt.plot(
            np.arange(0, sww_rx_subpulse.shape[1])
            * 1
            / TXRX_RATE,
            np.real(sww_rx_subpulse[0, :]),
        )
        plt.show()

    # turn off device after done
    total_time_end = time.time()
    print(f"Total time = {total_time_end-total_time_start}")

    print("Sending signal to stop TX and RX!")
    rx_stop_event.set()
    tx_stop_event.set()
    for thr in threads:
        thr.join()


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

    # create a test sensing plan list
    start_freq = 500e6
    stop_freq = 3e9
    freq_step = 10e6
    center_freq_list = np.arange(start_freq, stop_freq, freq_step)

    sensing_plan_list = []
    for _ in center_freq_list:
        center_freq = np.random.choice(center_freq_list)
        tx_baseband_signal, _ = complex_sinusoid(TXRX_RATE)
        tx_gains_list = [80, 80]
        num_rx_samps = tx_baseband_signal.shape[1] * 50
        rx_baseband_signal = np.zeros((2, num_rx_samps), dtype=np.complex64)
        rx_gains_list = [10, 10]

        sensing_plan = SensingPlan(
            center_freq,
            tx_baseband_signal,
            tx_gains_list,
            rx_baseband_signal,
            rx_gains_list,
        )

        sensing_plan_list.append(sensing_plan)

    # call the collector
    sww_data_collecter(usrp, tx_streamer, rx_streamer, sensing_plan_list)


if __name__ == "__main__":

    sys.exit(not main())