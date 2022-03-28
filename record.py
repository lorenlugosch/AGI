import pyaudio
import wave
import pexpect.replwrap
import subprocess
from datetime import datetime
import time
import mss
import numpy as np
import cv2
import matplotlib.pyplot as plt

INPUT_TAS_FILENAME = "timers-and-such/train-real/005360f2-b5ad-4bf5-b21b-ad3e9e06dbf1_prompt-181_0.wav" # the name of the Timers and Such audio file
OUTPUT_FILENAME = "prompt-181_0"
PROMPT = "'start timer for 19 seconds'"
print("PROMPT: " + PROMPT)

# Open adb shell
# (if this hangs, we're not root --- run "adb root")
adb_shell_process = pexpect.replwrap.REPLWrapper("adb shell", "#", None)
adb_shell_process.run_command("cd /data/local")

# play command
_ = subprocess.Popen("play " + INPUT_TAS_FILENAME + " -q", shell=True, stdout=subprocess.PIPE)

# start recording gestures
RAW_GESTURE_OUTPUT_FILENAME = OUTPUT_FILENAME + ".gestures"
adb_shell_process.run_command("cat /dev/input/event2 > " + RAW_GESTURE_OUTPUT_FILENAME + " &")
adb_shell_process.run_command("GESTURE_RECORD_PID=$!")
# gesture_poll_process = subprocess.Popen("adb shell cat /dev/input/event2", shell=True, stdout=subprocess.PIPE)

# record audio, gestures, screen until Enter
audio_frames = []
screen_frames = []
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
import _thread
def input_thread(done):
    input()             # use input() in Python3
    done.append(True)

done = []; _thread.start_new_thread(input_thread, (done,))
frame = 0
gesture_frames = []
timestamps = []
with mss.mss() as sct:
    emulator_screen = {"top": 120, "left": 900, "width": 300, "height": 750-120}

    # start recording microphone (for synchronization purposes)
    CHUNK = 1600 #CHUNK = 1024;
    FORMAT = pyaudio.paInt16; CHANNELS = 1; RATE = 16000;
    WAVE_OUTPUT_FILENAME = OUTPUT_FILENAME + ".wav"
    p = pyaudio.PyAudio(); stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* recording (hit Enter to stop)")

    while not done:
        # get screenshot
        # adb_shell_process.run_command("screencap " + str(frame) + ".raw &")
        img = np.array(sct.grab(emulator_screen))
        screen_frames.append(img[::2, ::2])

        # get audio
        data = stream.read(CHUNK)
        audio_frames.append(data)

        # get gestures (no command, this runs in GESTURE_RECORD_PID)
        #out = gesture_poll_process.stdout.read()
        #print(out)
        timestamps.append(time.time())
        frame += 1

    # stop recording audio
    print("* done recording")
    stream.stop_stream(); stream.close(); p.terminate()

# stop recording gestures, pull from phone
adb_shell_process.run_command("kill $GESTURE_RECORD_PID")
_ = subprocess.Popen("adb pull /data/local/" + RAW_GESTURE_OUTPUT_FILENAME, shell=True)

# save screen file
## file sizes for "set timer for 19 seconds":
## uncompressed             :  74 MB
## uncompressed --> zip -9  :   3 MB
## encode_sparse            :  13 MB
## encode_sparse --> zip -9 : 232 KB
##                      mp4 : 262 KB
## (encode_sparse is lossless and slightly smaller,
## but unzip is slower than mpeg decoding, and mp4
## can be viewed without unzipping and reconstructing)
SCREEN_OUTPUT_FILENAME = OUTPUT_FILENAME + ".mp4"
FPS = 10.0
codec = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(SCREEN_OUTPUT_FILENAME, codec, FPS, (300, 630)) #, isColor=True)
for frame in screen_frames:
    out.write(frame[:,:,:3]) # last channel not allowed
out.release()
## to read:
# cap = cv2.VideoCapture(SCREEN_OUTPUT_FILENAME)
# frames = []
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if not ret: break
#     frames.append(frame)


# save audio file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS); wf.setsampwidth(p.get_sample_size(FORMAT)); wf.setframerate(RATE)
wf.writeframes(b''.join(audio_frames))
wf.close()

# save gesture file
GESTURE_OUTPUT_FILENAME = OUTPUT_FILENAME + "_gestures.npy"
for i in range(len(timestamps)):
    timestamps[i] -= timestamps[0]

import struct
with open( "prompt-181_0.gestures", "rb" ) as f:
    FORMAT = '2IHHi'
    EVENT_SIZE = struct.calcsize(FORMAT)
    events = []
    while 1:
        try:
            data = f.read(EVENT_SIZE)
            events.append(struct.unpack(FORMAT,data))
        except:
            break

# NOOP = 0; DOWN = 1; UP = 2
# unaligned_gestures = []
# i = 0
# while i < len(events):
#     type = DOWN
#     location_x = -1
#     location_y = -1
#     current = events[i]
#     while current[2] != 0:
#         print(current)
#         if current[3] == 53: location_x = current[4]
#         if current[3] == 54: location_y = current[4]
#         if current[4] == -1:
#             type = UP
#             i += 1
#             break
#         i += 1
#         current = events[i]
#     location = (location_x, location_y)
#     timestamp = current[0] + current[1]/1000000
#     unaligned_gestures.append((timestamp, type, location))
#
# gestures = [(NOOP, (-1, -1)) for _ in range(len(screen_frames))]


print("Gestures saved to " + GESTURE_OUTPUT_FILENAME)
print("Audio saved to " + WAVE_OUTPUT_FILENAME)
print("Screenshots saved to " + SCREEN_OUTPUT_FILENAME)


#########
## from https://python-mss.readthedocs.io/examples.html#opencv-numpy
# import time
# import cv2
# import mss
# import numpy
# with mss.mss() as sct:
#     # Part of the screen to capture
#     monitor = {"top": 120, "left": 900, "width": 300, "height": 750-120}
#     while "Screen capturing":
#         last_time = time.time()
#         # Get raw pixels from the screen, save it to a Numpy array
#         img = numpy.array(sct.grab(monitor))
#         # Display the picture
#         # cv2.imshow("OpenCV/Numpy normal", img)
#         # Display downsampled picture
#         cv2.imshow("OpenCV/Numpy normal", img[::4, ::4])
#         # Display the picture in grayscale
#         # cv2.imshow('OpenCV/Numpy grayscale',
#         #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
#         print("fps: {}".format(1 / (time.time() - last_time)))
#         # Press "q" to quit
#         if cv2.waitKey(25) & 0xFF == ord("q"):
#             cv2.destroyAllWindows()
#             break

# a non-mp4 way to compress:
# def encode_sparse(screen_frames):
#     diff_indices = []
#     diffs = []
#     first = screen_frames[0]
#     prev = first
#     for frame in screen_frames[1:]:
#         diff_index = np.nonzero(prev != frame)
#         diff = frame[diff_index] - prev[diff_index]
#         diff_indices.append(diff_index)
#         diffs.append(diff)
#         prev = frame
#     return first, diff_indices, diffs
#
# def decode_sparse(encoded_video):
#     first, diff_indices, diffs = encoded_video
#     screen_frames = [first]
#     prev = first
#     for i in range(len(diffs)):
#         frame = prev.copy()
#         diff = diffs[i]
#         diff_index = diff_indices[i]
#         frame[diff_index] = prev[diff_index] + diff
#         screen_frames.append(frame)
#         prev = frame
#     return screen_frames
