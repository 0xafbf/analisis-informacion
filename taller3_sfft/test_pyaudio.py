import pyaudio
import time

import numpy as np
from bokeh.io import curdoc
from bokeh.plotting import Figure, ColumnDataSource
from bokeh.layouts import column
from bokeh.client import push_session
# from bokeh.driving import linear

from scipy.io import wavfile

sample_rate = 44100
timestep = 1 / sample_rate

t = np.arange(0, 1024)
y = np.sin(t)

f = np.fft.rfftfreq(y.size, timestep)
fft = np.absolute(np.fft.rfft(y))

sound_data = ColumnDataSource(data=dict(t=t, y=y))
fft_data = ColumnDataSource(data=dict(f=f, fft=fft))


def update():
    global f, fft
    f = np.fft.rfftfreq(y.size, timestep)
    # fft = np.absolute(np.fft.rfft(y))

    # fft[fft < FOURIER_CUTOFF] = 0

    sound_data.data["t"] = t
    sound_data.data["y"] = y
    fft_data.data["f"] = f
    fft_data.data["fft"] = np.absolute(fft)


figure_limit = (-35000, 35000)  # i16 min to i16 max
sound_figure = Figure(plot_width=1024, plot_height=400, y_range=figure_limit)
sound_figure.line(x="t", y="y", source=sound_data)

fft_figure = Figure(plot_width=1024, plot_height=400, y_range=(0, 800000))
fft_figure.line(x="f", y="fft", source=fft_data)

document = curdoc()
document.add_root(column(sound_figure, fft_figure))
session = push_session(document)
session.show()

# Refresh rate set to 30 hz
document.add_periodic_callback(update, 1000 / 30)

pa = pyaudio.PyAudio()

use_sound_file = True
input_file = "sax.wav"

fs, wav = wavfile.read(input_file)
wav = wav[:, 0]
# wav = wav.astype(np.float32, order='C') / 32768.0
wav = wav.astype(np.int16, order='C')

head = 0
buff_size = 1024

sample_rate = fs
timestep = 1 / sample_rate
PA_FORMAT = pyaudio.paInt16 if wav.dtype == np.int16 else pyaudio.paFloat32

FOURIER_CUTOFF = 8e4  # try 2k to 20k


def compress_rle(in_bytes, encode_element):
    count = 0
    output = bytearray()
    for b in in_bytes:
        if count == 255:
            output.append(encode_element)
            output.append(255)
            count = 0

        if b == encode_element:
            count += 1
        else:  # if the new element should not be encoded
            if count > 0:
                output.append(encode_element)
                output.append(count)
                count = 0

            output.append(b)

    if count > 0:
        output.append(encode_element)
        output.append(count)
        count = 0
    return bytes(output)


def uncompress_rle(in_bytes, encode_element):
    output = bytearray()
    for idx, b in enumerate(in_bytes):
        if b is encode_element:
            continue
        if idx == 0:
            output.append(b)
        elif in_bytes[idx - 1] == encode_element:
            output.extend([encode_element] * b)
        else:
            output.append(b)

    return bytes(output)


def callback(in_data, frame_count, time_info, flag):

    global y, t, head, fft
    data = None

    if use_sound_file:
        data = wav[head:head + buff_size]
        head += buff_size
        y = data
        t = np.arange(y.size)

        fft_temp = np.fft.rfft(y)

        # encode

        fft_abs = np.absolute(fft_temp)
        fft_temp[fft_abs < FOURIER_CUTOFF] = np.complex(0)

        fft_temp = fft_temp.view(np.float64)

        fft_temp = fft_temp / 8000

        fft_temp = fft_temp.astype(np.int16)

        bytes_data = fft_temp.tobytes()

        bytes_compressed = compress_rle(bytes_data, 0)

        bytes_uncompressed = uncompress_rle(bytes_compressed, 0)

        len_data = len(bytes_data)
        len_compressed = len(bytes_compressed)
        ratio = len_compressed / len_data

        print("len data: {}, Compressed: {},  final size: {}", len_data,
              len_compressed, ratio)

        fft_temp = np.frombuffer(bytes_uncompressed, np.int16)
        print("fft_temp shape", fft_temp.shape)
        # decode
        fft_temp = fft_temp.astype(np.float64)

        fft_temp = fft_temp * 8000
        fft_temp = fft_temp.view(np.complex128)

        fft = fft_temp

        y = np.fft.irfft(fft_temp, buff_size)
        data = y.astype(np.int16, order='C')

    else:
        y = np.fromstring(data, dtype=wav.dtype)
        t = np.arange(y.size)

    return (data, pyaudio.paContinue)


default_device = pa.get_default_output_device_info()

stream = pa.open(
    format=PA_FORMAT,
    channels=1,
    rate=sample_rate,
    output=use_sound_file,
    input=~use_sound_file,
    stream_callback=callback,
    output_device_index=default_device['index'])

stream.start_stream()

session.loop_until_closed()

while stream.is_active():
    time.sleep(0.25)
stream.close()
pa.terminate()
