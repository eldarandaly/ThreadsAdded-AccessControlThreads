import socket, os,cv2, pickle,struct,time,imutils
import numpy as np
import threading as thread
import threading
from DoorDistance import distance
import RPi.GPIO as GPIO
#from demo_lcd import ShowonLcd
from time import sleep
import time 
import drivers

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.output(4, 1)
state =True
display = drivers.Lcd()
screen_signal=''
start = 0
doorUnlock = False # DOOR LOCK FLAG
GPIO.setup(13,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
buzzer=20
GPIO.setup(buzzer,GPIO.OUT)
# tcp and ipv4 address family
tcp = socket.SOCK_STREAM
afm = socket.AF_INET
#Camera
gate_camera='http://192.168.0.4:8080/video'
# Cleint IP/port
user_a_ip = "192.168.1.62"
user_a_port = 9000

#Server IP/port
user_b_ip = "192.168.1.2"
user_b_port = 9001
# creating socket
sa = socket.socket(afm,tcp)
sb = socket.socket(afm,tcp)

# Binding ports 
sa.bind((user_a_ip,user_a_port))

# listening port and creating session
sa.listen(5)
display.lcd_display_string("Wating   For", 1)
display.lcd_display_string("    Connection", 2)
print("Wating For Connection")
session, addr = sa.accept()


print(addr)

# connecting to userb 
sb.connect((user_b_ip,user_b_port))
display.lcd_clear()
display.lcd_display_string("Conntected", 1)
#display.lcd_clear()
class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes
        self.callback = None

        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            
            assert rv
            counter += 1

            # publish the frame
            with self.cond:  # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self.cond.wait_for(
                    lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)
def receive():
    #screen_signal=''
    comm_flag=False
    while session:
        sig = session.recv(1)
        Signal=sig.decode("utf-8")
        if type(Signal) is type(None):
            pass
        print(Signal)
        state= GPIO.input(13)
        print(state)
       
        if Signal == "o" :
                # to unlock the door
                #print("writing to display")
                #display.lcd_clear()
                #display.lcd_display_string("Access Confrimed", 1)  # Write line of text to first line of display
                doorUnlock = True
                GPIO.output(4, 0)
                #sleep(1)
                screen_signal='o'
                #send_thread = thread.Thread(target=screen_Text,args=('o',))
                print("door unlock")
                #display.lcd_clear()
                #display.lcd_display_string("Close The Door", 2)
                #sleep(1)
                if (doorUnlock == True):
                    doorUnlock = False
                    GPIO.output(4, 1)
                    print("door lock")
                    #comm_flag==True
                    #Signal=''
                    
        elif Signal == "x" :
            print("DontOpen")
            screen_signal='x'
            #send_thread = thread.Thread(target=screen_Text,args=('x',))
            #send_thread.start()
           # display.lcd_clear()
            #display.lcd_display_string("No Access", 1)
            #sleep(1)
            Signal=''
            # display.lcd_clear()
            #pass
        elif Signal == "t":
            print("Two Person")
            
            screen_signal='t'

            #send_thread = thread.Thread(target=screen_Text,args=('t',))
            #send_thread.start()
            #display.lcd_clear()
            #display.lcd_display_string("Two Faces", 1)
            #display.lcd_display_string("detecteced",2)
            ##sleep(1)
            #Signal='' 
            #pass
        
        elif (state == 1): # state == 1 when you press your hand 
            print("The Door is opend ")
            GPIO.output(4, 0)
            screen_signal='b'

            #send_thread = thread.Thread(target=screen_Text,args=('b',))
            #send_thread.start()# open the door lock
            #display.lcd_clear()
            #display.lcd_display_string("Door Opened", 1)
            #display.lcd_display_string("From Inside", 2)
            #sleep(1)
            # time.sleep(5)
            GPIO.output(4,1) #close
        elif Signal == 'u':
            print("unknow")
            screen_signal='u'

            #send_thread.start()
            #display.lcd_clear()
            #display.lcd_display_string("---UNKNOW---",1)
            #display.lcd_display_string("Center your Face",2)
            #sleep(1)
            #Signal=''

        elif Signal == "a":
            screen_signal='a'

            #display.lcd_clear()
            #display.lcd_display_string("No Face Detected",1)
            #sleep(1)
            #send_thread = thread.Thread(target=screen_Text,args=(screen_signal,))
            #send_thread.start()
            print("no face ")
            #Signal=''
            # print("noface")    
        #elif Signal == '':
            #pass
            #sleep(2)
            #sb.close()
            #exit(-1)
        elif Signal=='q':
            break
            #sa.close()
            #sb.close()
            #display.lcd_clear()
            #GPIO.cleanup()
            #exit(-1)
            
        #screen_Text(screen_signal)
        #send_thread = thread.Thread(target=screen_Text,args=(screen_signal,))
        #send_thread.start()
        
        #display.lcd_display_string("Point Your Face", 1)
        #display.lcd_display_string("To The Camera", 2)
    sa.close()
    sb.close()
    display.lcd_clear()
    GPIO.cleanup()
    exit(-1)
            
'''def screen_Text(screen_signal):    
    #while True:
        display.lcd_clear()
        sleep(0.5)
        if screen_signal == "o":
                display.lcd_clear()
                
                display.lcd_display_string("Access Confrim", 1)  # Write line of text to first line of display
                sleep(2)
                display.lcd_display_string("Close The Door", 2)
                sleep(2)
                screen_signal=''
                    
        elif screen_signal == "x":
            print("DontOpen")
            display.lcd_clear()
            display.lcd_display_string("No Access", 1)
            sleep(2)
            screen_signal=''
            # display.lcd_clear()
            #pass
        elif screen_signal == "t":
            print("Two Person")
            display.lcd_clear()
            display.lcd_display_string("Two Person", 1)
            display.lcd_display_string("One face at Time",2)
            screen_signal=''
            sleep(2)
            #pass
        
        elif (screen_signal == 'b'): # state == 1 when you press your hand 
            display.lcd_clear()
            display.lcd_display_string("Door Opened", 1)
            display.lcd_display_string("From Inside", 2)
            screen_signal=''
            sleep(2)

        elif screen_signal == 'u':
            display.lcd_clear()
            display.lcd_display_string("UNKNOW",1)
            screen_signal=''
            sleep(2)

        elif screen_signal == "a":
            display.lcd_clear()
            display.lcd_display_string("No Face Detected",1)
            screen_signal=''
            sleep(2)
        elif screen_signal=='':
            pass
        display.lcd_clear()  
         
def send():
    capture = cv2.VideoCapture(gate_camera)
    fresh=FreshestFrame(capture)
    while(True):
        img,frame = fresh.read()
        frame = imutils.resize(frame,width=320)
        #frame= cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        a = pickle.dumps(frame)
        message = struct.pack("Q",len(a))+a
        sb.sendall(message)

def distanceSensor():
    while True:
         dist = distance()
         print ("Measured Distance = %.1f cm" % dist)
         if dist < 7 :
             GPIO.output(buzzer,GPIO.HIGH)
             
         else:
             GPIO.output(buzzer,GPIO.LOW)
         time.sleep(1)
 #   os.exit_(1) '''    
# send and receive threads
#send_thread = thread.Thread(target=screen_Text,args=(screen_signal,))
#send_thread.start()
recv_thread = thread.Thread(target=receive)
#dist_thread = thread.Thread(target=distanceSensor)
# starting threads
screen_signal=''
recv_thread.start()
#dist_thread.start()
