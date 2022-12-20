adb shell sendevent /dev/input/event2 3 57 0
adb shell sendevent /dev/input/event2 3 53 5651
adb shell sendevent /dev/input/event2 3 54 17061
adb shell sendevent /dev/input/event2 3 58 1
adb shell sendevent /dev/input/event2 0 0 0

adb shell sendevent /dev/input/event2 3 57 -1
adb shell sendevent /dev/input/event2 0 0 0

######
# from https://stackoverflow.com/questions/1908610/how-to-get-process-id-of-background-process
# in a writeable directory,
# launch screenrecord, get PID
# launch gesture record, get PID
# kill screenrecord
# kill gesture record

cat /dev/input/event2 > gesture &
GESTURE_RECORD_PID=$!
<record screen command> &
SCREEN_RECORD_PID=$!

kill $GESTURE_RECORD_PID
kill $SCREEN_RECORD_PID

# idea: resynthesize episodes
# by varying the speed/position of gestures
# and recording the screen as the gestures play out


out = p.run_command("ls")
_ = p.run_command("cd /data/local")
out = p.run_command("ls")

# loop:
# 1. get audio from pyaudio
# 2. get screenshot from adb shell
# 3. add audio and screenshot to buffer
# 4. run policy on buffer
#

import subprocess
import time
with subprocess.Popen("adb shell", stdin = subprocess.PIPE, stdout=subprocess.PIPE, shell=True) as p:
  time.sleep(1)
  p.stdin.write("ls".encode())
