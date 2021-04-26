from gpiozero import Button
from picamera import PiCamera
from time import sleep
import os, sys
#import cv2

def get_frames():
  cap = cv2.VideoCapture('/home/pi/Desktop/video.h264')
  i =0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    cv2.imwrite('test_'+str(i)+'.jpg',frame)
    i += 1
  cap.release()
  cv2.destroyAllWindows()

os.system('omxplayer -o local /home/pi/xWALK/pifiles/Ready.wav');

button = Button(10)

while(1):  
  button.wait_for_press()
 
  camera=PiCamera()
  sleep(2) 
  camera.capture('/home/pi/Desktop/image.jpg')
  #camera.stop_recording()
  camera.close()

#  get_frames()
  os.system('omxplayer -o local /home/pi/xWALK/pifiles/Done.wav')

print("Button was pressed")
