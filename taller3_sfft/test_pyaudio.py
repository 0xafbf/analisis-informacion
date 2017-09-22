import pyaudio
import time

import numpy as np
from bokeh.io import curdoc
from bokeh.plotting import Figure, ColumnDataSource
from bokeh.layouts import column
from bokeh.client import push_session
# from bokeh.driving import linear


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
    fft = np.absolute(np.fft.rfft(y))

    sound_data.data["t"] = t
    sound_data.data["y"] = y
    fft_data.data["f"] = f
    fft_data.data["fft"] = fft


figure_limit = (-0.5, 0.5)
sound_figure = Figure(plot_width=1024, plot_height=400, y_range=figure_limit)
sound_figure.line(x="t", y="y", source=sound_data)

fft_figure = Figure(plot_width=1024, plot_height=400, y_range=(0, 200))
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

from scipy.io import wavfile
fs, wav = wavfile.read(input_file)
wav = wav[:, 0]
wav = wav.astype(np.float32, order='C') / 32768.0
head = 0
buff_size = 1024

print(wav)
sample_rate = fs
timestep = 1 / sample_rate


def callback(in_data, frame_count, time_info, flag):
    data = None

    global y, t, head

    if use_sound_file:
        data = wav[head:head + buff_size]
        head += buff_size
        y = data
        t = np.arange(y.size)

    else:
        y = np.fromstring(data, dtype=np.float32)
        t = np.arange(y.size)

    return (data, pyaudio.paContinue)


stream = pa.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=sample_rate,
    output=use_sound_file,
    input=~use_sound_file,
    stream_callback=callback)

stream.start_stream()

session.loop_until_closed()

while stream.is_active():
    time.sleep(0.25)
stream.close()
pa.terminate()
