from gpiozero import Button
from picamera import PiCamera
from time import sleep
import pygame

pygame.mixer.pre_init(44100,16,2,4096)
pygame.mixer.init()




pygame.mixer.music.load('xWALK/pifiles/Ready.wav')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
  continue

button = Button(10)
camera=PiCamera()
button.wait_for_press()
camera.start_recording('/home/pi/Desktop/video.h264')




sleep(3)

camera.stop_recording()
camera.close()
print("Button was pressed")
