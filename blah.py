import subprocess
import threading
import time
import queue

# from https://eli.thegreenplace.net/2017/interacting-with-a-long-running-child-process-in-python/
# also worth noting? https://stackoverflow.com/questions/11689511/transferring-binary-data-over-adb-shell-ie-fast-file-transfer-using-tar?noredirect=1&lq=1
def output_reader(proc, outq):
    for line in iter(proc.stdout.readline, b''):
        #print('got line: {0}'.format(line.decode('utf-8')), end='')
        outq.put(line)

#proc = subprocess.Popen("adb shell 'cat /dev/input/event2'", shell=True, stdout=subprocess.PIPE)
proc = subprocess.Popen("adb shell 'od -x /dev/input/event2'", shell=True, stdout=subprocess.PIPE)
outq = queue.Queue()
t = threading.Thread(target=output_reader, args=(proc,outq))
t.start()

while True:
    try:
        line = outq.get(block=False)
        print(line)
    except queue.Empty:
        continue
#     time.sleep(0.1)
