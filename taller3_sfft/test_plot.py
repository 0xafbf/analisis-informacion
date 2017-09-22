import numpy as np
from bokeh.io import curdoc
from bokeh.plotting import Figure, ColumnDataSource
from bokeh.client import push_session

from bokeh.driving import linear

import pyaudio

x = np.linspace(0, 10, 100)
y1 = np.sin(x)

myfigure = Figure(plot_width=800, plot_height=400)
datacoords = ColumnDataSource(data=dict(x=x, y1=y1))
myfigure.line(x="x", y="y1", source=datacoords)
myfigure.line(x="x", y="y2", source=datacoords)


@linear(m=0.05, b=0)  # step will increment by 0.05 every time
def update(step):

    print("bokeh update")
    new_x = np.linspace(step, step + 10, 100)
    new_y1 = np.sin(new_x)
    new_x, new_y1 = x, y1

    new_y2 = np.cos(new_x)

    datacoords.data["x"] = new_x
    datacoords.data["y1"] = new_y1
    datacoords.data["y2"] = new_y2


document = curdoc()
document.add_root(myfigure)
session = push_session(document)
session.show()
document.add_periodic_callback(update, 10)  # period in ms


# instantiate PyAudio (1)
pa = pyaudio.PyAudio()


# define callback (2)
def callback(in_data, frame_count, time_info, status):
    print("pa_callback")
    # convert data to array
    y1 = np.fromstring(in_data, dtype=np.float32)
    global x
    x = np.arange(y1.size)
    # process data array using librosa
    # ...
    return (None, pyaudio.paContinue)


# open stream using callback (3)
stream = pa.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=44100,
    input=True,
    output=False,
    frames_per_buffer=int(44100 * 10),
    stream_callback=callback)

# start the stream (4)
stream.start_stream()


session.loop_until_closed()

# stop stream (6)
stream.stop_stream()
stream.close()

# close PyAudio (7)
pa.terminate()
