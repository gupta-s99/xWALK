from gpiozero import Button
from picamera import PiCamera
from time import sleep

button = Button(10)
camera=PiCamera()
button.wait_for_press()
camera.start_recording('/home/pi/Desktop/video.h264')
sleep(5)

camera.stop_recording()
camera.close()
print("Button was pressed")
