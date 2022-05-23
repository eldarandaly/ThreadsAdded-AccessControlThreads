import socket, os,cv2
import numpy as np
import threading as thread
from DoorDistance import distance
# import RPi.GPIO as GPIO
#from demo_lcd import ShowonLcd
from time import sleep
# import drivers
# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(4, GPIO.OUT)
# GPIO.output(4, 0)
state =True
# 
# GPIO.setup(13,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
# tcp and ipv4 address family
tcp = socket.SOCK_STREAM
afm = socket.AF_INET

# user a
usera_ip = "192.168.1.2"
usera_port = 2000

# creating socket
sa = socket.socket(afm,tcp)
sb = socket.socket(afm,tcp)

# Binding ports 
sa.bind((usera_ip,usera_port))

# listening port and creating session
sa.listen()
print("Listening")
session, addr = sa.accept()

print(addr)

# connecting to userb 
sb.connect((usera_ip,2001))

def receive():

    while True:
        sig = session.recv(1)
        Signal=sig.decode("utf-8")
        # image_arr = np.frombuffer(en_photo,np.uint8)
        # image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        if type(Signal) is type(None):
            pass
        # else:
        # print(Signal)
        if Signal == "o":
                # to unlock the door
                #start = time.time()
                #print("writing to display")
                # display.lcd_display_string("Access Confrimed", 1)  # Write line of text to first line of display
                # display.lcd_display_string("Ahmed", 2)
                doorUnlock = True
                # GPIO.output(4, 1)
                print("door unlock")
                # display.lcd_display_string("Close The Door", 1) 
                #print("Open")
                # sleep(5)
                if (doorUnlock == True):
                    doorUnlock = False
                    # GPIO.output(4, 0)
                    print("door lock")
                    
                    # display.lcd_clear()
                    # display.lcd_display_string("Close The Door", 1)
        elif Signal == "x":
            print("DontOpen")
            # display.lcd_display_string("No Access", 1)
            # display.lcd_clear()
            #pass
        elif Signal == "t":
            print("Two Person")
            # display.lcd_display_string("Alarm", 1)
            #pass
        
        # elif (state == 0): # state == 1 when you press your hand 
            # print("The Door is opend ")
            # GPIO.output(4, 1) # open the door lock 
            # time.sleep(5)
            # GPIO.output(4,0) #close
        elif Signal == 'u':
            print("unknow")
            # display.lcd_display_string("UNKNOW",1)
            # display.lcd_clear()
        # elif Signal == "z":
            # display.lcd_display_string("No Face Detected",1)
            # display.lcd_clear()
            # print("noface")    

    os._exit(1)

def send():
    capture = cv2.VideoCapture(0)

    while True:
        ret, photo = capture.read()
        if ret == True:
            en_photo = cv2.imencode('.jpg',photo)[1].tobytes()
            sb.sendall(en_photo)
        else: 
            pass
            
    os.exit_(1)

def distanceSensor():
    distance() 
# send and receive threads
send_thread = thread.Thread(target=send)
recv_thread = thread.Thread(target=receive)
dist_thread = thread.Thread(target=distanceSensor)
# starting threads
send_thread.start()
recv_thread.start()
dist_thread.start()