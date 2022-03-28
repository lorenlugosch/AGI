# adapted from https://thehackerdiary.wordpress.com/2017/04/21/exploring-devinput-1/

import struct 
f = open( "prompt-181_0.gestures", "rb" ); # Open the file in the read-binary mode
#f = open( "swipe", "rb" ); # Open the file in the read-binary mode
FORMAT = '2IHHi'
EVENT_SIZE = struct.calcsize(FORMAT)
print("event size = %d" % EVENT_SIZE)
while 1:
  data = f.read(EVENT_SIZE)
  print(struct.unpack(FORMAT,data))
  ###### PRINT FORMAL = ( Time Stamp_INT , 0 , Time Stamp_DEC , 0 , 
  ######   type , code ( key pressed ) , value (press/release) )
