import socket, cv2, os
from datetime import datetime
import numpy as np
import threading as thread
from time import sleep
from PIL import Image
from keras.models import model_from_json
# from tensorflow.keras.preprocessing.image import img_to_array
from tkinter import Tk     
from tkinter.filedialog import askopenfilename
from xlsxwriter. workbook import Workbook
import threading
import os,pickle,socket,struct,sqlite3,cv2
import numpy as np
import tensorflow as tf
import TFdetect_face as detect_face
import TFfacenet as facenet 
import pandas as pd
# tcp and ipv4 address family
tcp = socket.SOCK_STREAM
afm = socket.AF_INET
#dfs


ThisGate='2'

# Server Ip/port
user_b_ip = '192.168.1.2'
user_b_port = 9001

#Client IP/port
# client_ip=input("Please Enter The Gate IP like [192.168.xxx.xxx] or Enter Gate ID [1,2,3,...] ")
user_a_ip =  '192.168.1.62'
user_a_port = 9000

# creating socket
sb = socket.socket(afm,tcp)
sa = socket.socket(afm,tcp)

# bindiing ports 
sb.bind((user_b_ip,user_b_port))

# connecting to usera
sa.connect((user_a_ip,user_a_port))

# listening port and creating session
sb.listen()
session, addr = sb.accept()

print(addr)
"""--------------------------------This is The ServerCode for our MTCNN Model-----------------------------"""
data = b""
payload_size = struct.calcsize("Q")

modeldir = 'PCServer/model/VGGFaces.pb'
classifier_filename = 'PCServer/class/TEST_IP.pkl'
npy='PCServer/npy'
train_img="PCServer/training/TrainFolder50imgPerClass"

configPath = 'PCServer/PersonDetectionModel/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # tf
weightsPath = 'PCServer/PersonDetectionModel/frozen_inference_graph.pb'

thres = 0.6 # Threshold to detect object
nms_threshold = 0.45
path_to='C:/Users/ahmed/OneDrive/Desktop'
DATABASEPATH = path_to+'/ProjectDatabase.db'
# DATABASEPATH="PCServer/NewSeniorDataBase.db"

classNames= []
Personfile = 'PCServer/PersonDetectionModel/coco.names'

sendSignal='j'
firstRunimg='PCServer/training/TrainFolder50imgPerClass/Ahmed.1/img (1).jpeg'
ENTER_CAMERA=0#'http://192.168.1.44:8080/video' #"rtsp://admin:TZZUNI@192.168.1.58:554/H.264"#0#'http://192.168.1.44:8080/video'

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
            try:
                (rv, img) = self.capture.read()

                assert rv
                counter += 1
            except Exception:
                print("no camera detected")
                continue

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

def receive(sendSignal):
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess: 

            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size = 4  # 1000
            image_size = 182  # 182
            input_image_size = 160  # 160
            
            EmployeeModelNames = os.listdir(train_img)
            EmployeeModelNames.sort()

            real_face_list = []
            fake_face_list = []
            
            print('Loading Model')
            json_file = open('PCServer/antispoofing_models/antispoofing_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            antiSpofingmodel = model_from_json(loaded_model_json)
            antiSpofingmodel.load_weights('PCServer/antispoofing_models/antispoofing_model.h5')
            print("AntiSpoofing Model Loaded")
            facenet.load_model(modeldir)

            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')
          
            with open(Personfile,'rt') as f:
                classNames = f.read().rstrip('\n').split('\n')

            net = cv2.dnn_DetectionModel(weightsPath,configPath) 
            net.setInputSize(320,320)
            net.setInputScale(1.0/ 127.5) #INFO 255/2 = 127.5
            net.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1, 1]
            net.setInputSwapRB(True)
            
            '''# while True:
            #     en_photo = session.recv(921600)
            #     image_arr = np.frombuffer(en_photo,np.uint8)
            #     frame = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
            #     if type(frame) is type(None):
            #         pass
                # else:
                    # cv2.imshow("Video stream", frame)
                    # if cv2.waitKey(10) == 13: 
                        # break

                # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                # frame=cv2.flip(frame,0)'''
            data = b""
            payload_size = struct.calcsize("Q")
            try:
                First_Run(sess, pnet, rnet, onet, minsize, threshold, factor, image_size, input_image_size,
                        antiSpofingmodel, images_placeholder, embeddings, phase_train_placeholder, embedding_size, model)
            except:
                print("--pass Fitst run IMG")
            entering_camera = cv2.VideoCapture(ENTER_CAMERA)
            entering_camera_therad = FreshestFrame(entering_camera)
            if not (entering_camera.isOpened()):
                print("no camera detected")
            else:
                pass


            while session:
                ''' while len(data) < payload_size:
                #     packet = session.recv(4*1024) # 4K Socket Recive
                #     if not packet: break
                #     data+=packet
                # packed_msg_size = data[:payload_size]
                # data = data[payload_size:]
                # msg_size = struct.unpack("Q",packed_msg_size)[0]
                # while len(data) < msg_size:
                #     data += session.recv(4*4096)

                # frame_data = data[:msg_size]
                # data  = data[msg_size:]
                # frame = pickle.loads(frame_data)
                # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)'''

                ret, frame = entering_camera_therad.read()  # Entering Cam
                # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                try:
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)    
                    
                    classIds, confs, bbox = net.detect(frame,confThreshold=thres)
                    bbox = list(bbox)
                    confs = list(np.array(confs).reshape(1,-1)[0])
                    confs = list(map(float,confs))
                    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
                    isPerson=2
                    try:
                        if (indices.size ==0) or int(indices[0][0])== 0 :
                            isPerson=1
                        else:
                            isPerson=0
                    except:
                        pass

                    faceNum = bounding_boxes.shape[0]

                    '''
                    No face Detected and there are a Person in flied of view trying to Enter 
                    Trying To use this methode to prevent the access of people hide there face 

                    '''
                    if faceNum==0 and isPerson==1: 
                        for i in indices:
                            acc=float(confs[0])
                            if acc>0.7:
                                i = i[0]
                                box = bbox[i]
                                x,y,w,h = box[0],box[1],box[2],box[3]
                                whatIsit=classNames[classIds[i][0]-1]
                                # print(whatIsit)
                                if whatIsit == 'person':
                                    cv2.rectangle(frame, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
                                    cv2.putText(frame,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

                                    sendSignal='body' 
                                    # send_thread = thread.Thread(target=send,args=(sendSignal,))
                                    # send_thread.start()                                                
                                else:
                                    pass

                    elif faceNum ==1: 
                    
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]
                        cropped = []
                        scaled = []
                        scaled_reshape = []

                        for i in range(faceNum):

                            emb_array = np.zeros((1, embedding_size))
                            xmin = int(det[i][0])
                            ymin = int(det[i][1])
                            xmax = int(det[i][2])
                            ymax = int(det[i][3])
                        
                            try:
                                # inner exception
                                if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                    # print('Face is very close!')
                                    continue
                                cropped.append(frame[ymin:ymax, xmin:xmax,:])

                                cropped[i] = facenet.flip(cropped[i], False)

                                scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))

                                scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                        interpolation=cv2.INTER_CUBIC)

                                scaled[i] = facenet.prewhiten(scaled[i])

                                scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))

                                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}

                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)

                                best_class_indices = np.argmax(predictions, axis=1)

                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                # print(best_class_indices,best_class_probabilities)
                                face = frame[ymin:ymax, xmin:xmax]
                                try:
                                    
                                    resized_face = cv2.resize(scaled[i], (160, 160))
                                    cv2.imwrite("After_Detected_Croped_Face.jpg",resized_face)
                                    resized_face = resized_face.astype("float") / 255.0
                                    resized_face = np.expand_dims(resized_face, axis=0)
                                    preds = antiSpofingmodel.predict(resized_face)[0]
                                    # if preds>0.8:
                                    #     break
                                except:
                                    print("Spoofing cant predict")
                                    continue
                                ''' 
                                # pass the face ROI through the trained liveness detector
                                # model to determine if the face is "real" or "fake"
                                '''
                                preds = antiSpofingmodel.predict(resized_face)[0]
                                print("Spoofing Meter->",preds)
                                indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
                                try:
                                    if (indices.size ==0) or int(indices[0][0])== 0 :
                                        isPerson=1
                                        print("person detected")
                                    else:
                                        isPerson=0
                                        print("not person")
                                except:
                                    pass

                                if best_class_probabilities > 0.7 and preds < 0.4 and isPerson==1: # Real prson and have acc >> 70% and his Face detected 
                                    # if isPerson==0:  
                                    for i in indices:                     
                                        acc=float(confs[0])
                                        i = i[0]
                                        box = bbox[i]
                                        x,y,w,h = box[0],box[1],box[2],box[3]

                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                    for EmployeeNames in EmployeeModelNames:
                                        if EmployeeModelNames[best_class_indices[0]] == EmployeeNames:

                                            result_names = EmployeeModelNames[best_class_indices[0]]

                                            name  = result_names.split('.')[0]
                                            Id = result_names.split('.')[1]
                                            name=str(name)
                                            sendID=str(Id)
                                            
                                            check=checkIfHaveAccess(sendID,name)
                                            if Id not in real_face_list:
                                                real_face_list.append(Id)

                                                if check:
                                                    sendSignal='open'                                                 
                                                else:
                                                    sendSignal='dont'
                                                                        
                                            print("Predictions : [ name: {} , accuracy: {:.3f}  ,preds :{}]".format(EmployeeModelNames[best_class_indices[0]], best_class_probabilities[0],preds))
                                            label = 'Real'
                                            cv2.putText(frame, label, (xmax, ymax),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),(0, 0, 255), 2)
                                            # print("Predictions : [ name: {} , accuracy: {:.3f} ]".format( EmployeeModelNames[best_class_indices[0]], best_class_probabilities[0]))
                                            cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (255, 255, 255), -1)
                                            cv2.putText(frame, result_names, (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5, (0, 0, 0), thickness=1, lineType=1)

                                            cv2.rectangle(frame, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2) #person rect
                                            cv2.putText(frame,classNames[classIds[0][0]-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) #person text
   
                                                                                    
                                elif best_class_probabilities < 0.7 and best_class_probabilities > 0.6 and preds > 0.7 and isPerson==1: #Fake person and have acc> 70% but Spoofing 
                                    
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                    
                                    for EmployeeNames in EmployeeModelNames:
                                        if EmployeeModelNames[best_class_indices[0]] == EmployeeNames:
                                            
                                            result_names = EmployeeModelNames[best_class_indices[0]]
                                            name=result_names.split('.')[0]
                                            Id = result_names.split('.')[1]
                                            
                                            if Id not in fake_face_list:
                                                
                                                fake_face_list.append(Id)
                                                sendSignal='dont'
                                                # send_thread = thread.Thread(target=send,args=(sendSignal,))
                                                # send_thread.start() 

                                            label = 'Fake '+result_names+"is Taken as Spoofed"
                                            cv2.putText(frame, label, (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),(0, 0, 255), 2)
                                            print("Predictions : [ name: {} , accuracy: {:.3f}  ,preds :{}]".format(EmployeeModelNames[best_class_indices[0]], best_class_probabilities[0],preds))
                                            cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (255, 255, 255), -1)
                                            cv2.putText(frame, result_names, (xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5, (0, 0, 0), thickness=1, lineType=1)
 
                                    print("Employee spoof attack")
                                
                                elif preds > 0.7 and isPerson==0:
                                    cv2.rectangle(frame, (xmin, ymin),(xmax, ymax), (255, 0, 0), 2)
                                    for EmployeeNames in EmployeeModelNames:
                                        if EmployeeModelNames[best_class_indices[0]] == EmployeeNames:
                                                result_names = EmployeeModelNames[best_class_indices[0]]

                                                name = result_names.split('.')[0]
                                                Id = result_names.split('.')[1]

                                                label = 'Fake'
                                                sendSignal='fake'
                                                
                                                cv2.putText(frame, label, (xmax, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                                cv2.rectangle(frame, (xmin, ymin),(xmax, ymax), (255, 0, 0), 2)

                                                print("Predictions : [ name: {} , accuracy: {:.3f}  ,preds :{}]".format(EmployeeModelNames[best_class_indices[0]], best_class_probabilities[0],preds))



                                else : # face detected but Unkonw 
                                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                    cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                    cv2.putText(frame, "Unknow", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 0), thickness=1, lineType=1)
                                    sendSignal='unkn'
                                    # send_thread = thread.Thread(target=send,args=(sendSignal,))
                                    # send_thread.start()   
                            except:  
                                print("[ERROR] Frame skipped ")

                    elif faceNum>1: # Person Detected but No Face Detected 
                        sendSignal='t'

                    cv2.imshow('Face Recognition', frame)
                    
                    try:
                        if sendSignal == "open": # Have Access
                            sa.sendall(bytes("open","utf-8"))                            
                            print("sended",sendSignal)
                            sendSignal='Null' # reset 

                        if sendSignal == 'dont': # dont have access or spoofing
                            sa.sendall(bytes("dont","utf-8"))
                            print("sended",sendSignal)
                            sendSignal='Null'
                            
                        if sendSignal == 'fake': #  spoofing
                            print("sended",sendSignal)
                            sa.sendall(bytes("fake","utf-8"))
                            sendSignal='Null'

                        if sendSignal == 'body':
                               # person and no face detected 
                            print("sended",sendSignal)
                            sa.sendall(bytes("body","utf-8"))
                            sendSignal='Null'

                        if sendSignal == 'twoo': # Two Person Detected
                            sa.sendall(bytes("twoo","utf-8"))
                            print("sended",sendSignal)
                            sendSignal='Null'

                        if sendSignal == 'unkn':
                             # unknow                      
                            sa.sendall(bytes("unkn","utf-8"))
                            print("sended",sendSignal)
                            sendSignal='Null'

                        #sleep(1)
                        real_face_list.clear()
                        fake_face_list.clear()

                        sendSignal='N' # reset 
                        # sa.sendall(bytes('\x00',"utf-8"))
                        # sa.sendall(bytes(sendSignal,"utf-8"))
                        # sendSignal='N' # reset 

                    except socket as Error:
                        print("[ERROR] cant send ")

                    key = cv2.waitKey(1) & 0xFF
                    if key  == ord('q'):
                        sa.sendall(bytes("q","utf-8"))
                        break
                except:
                    print("Skipped")
                    pass
            cv2.destroyAllWindows()
            sa.close()
            sb.close()
            os._exit(1)

''' def send(sendSignal):

#     if sendSignal == "o": # Have Access
#         sa.sendall(bytes("o","utf-8"))
#         # sendSignal='Null' # reset 

#     if sendSignal == 'x': # dont have access or spoofing
#         sa.sendall(bytes("x","utf-8"))

#     if sendSignal == 'a': # person and no face detected 
#         sa.sendall(bytes("a","utf-8"))

#     if sendSignal == 't': # Two Person Detected
#         sa.sendall(bytes("t","utf-8"))

#     if sendSignal == 'u':  # unknow                      
#         sa.sendall(bytes("u","utf-8"))
#         sendSignal='N' # reset 
#         # sa.sendall(bytes('\x00',"utf-8"))
#         # sa.sendall(bytes(sendSignal,"utf-8"))
#         # sendSignal='N' # reset'''


def First_Run(sess, pnet, rnet, onet, MIN_SIZE, THERSHOLD, factor, IMAGE_SIZE, INPUT_IMAGE_SIZE, antiSpofingmodel, images_placeholder, embeddings, phase_train_placeholder, embedding_size, model):
    frame = cv2.imread(firstRunimg)
    bounding_boxes, _ = detect_face.detect_face(frame, MIN_SIZE, pnet, rnet, onet, THERSHOLD, factor)
    faceNum = bounding_boxes.shape[0]

    if faceNum > 0:  # Number of faces on the screen
        det = bounding_boxes[:, 0:4]  # get x ,y,width,height

        img_size = np.asarray(frame.shape)[0:2]  # get image size
        cropped = []
        scaled = []
        scaled_reshape = []

        for i in range(faceNum):
            # get Embedding of the detected face
            emb_array = np.zeros((1, embedding_size))

            # points around the detected face
            xmin = int(det[i][0])
            ymin = int(det[i][1])  # ||  ||
            xmax = int(det[i][2])  # ||  ||
            ymax = int(det[i][3])  # ||  ||

            try:
                # inner exception if the face is to close to the camear
                if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                    print('Face is very close!')
                    continue
                    # crop the face to recog
                cropped.append(frame[ymin:ymax, xmin:xmax, :])

                # Mirror the cropped face img
                cropped[i] = facenet.flip(cropped[i], False)

                scaled.append(np.array(Image.fromarray(cropped[i]).resize(
                    (IMAGE_SIZE, IMAGE_SIZE))))  # scale the cropped img

                scaled[i] = cv2.resize(
                    scaled[i], (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

                scaled[i] = facenet.prewhiten(scaled[i])

                scaled_reshape.append(
                    scaled[i].reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))

                feed_dict = {
                    images_placeholder: scaled_reshape[i], phase_train_placeholder: False}

                # GET Between the detected Embeddings and the preTrained Model
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                # Predict Between the detected Embeddings and the preTrained Model
                predictions = model.predict_proba(emb_array)

                # get the Best Recog Class label
                best_class_indices = np.argmax(predictions, axis=1)

                # get the acc
                best_class_probabilities = predictions[np.arange(
                    len(best_class_indices)), best_class_indices]

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                face = frame[ymin:ymax, xmin:xmax]
                resized_face = cv2.resize(face, (160, 160))
                resized_face = resized_face.astype("float") / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)

                preds = antiSpofingmodel.predict(resized_face)[0]
                print("LoadingDone")
            except:
                print("error")

def checkIfHaveAccess(EmpID,emp_name,IsSpoofing):

    dbflag=True

    try:
        conn = sqlite3.connect(DATABASEPATH) # connected to Raspberrypi 4 "//raspberrypi/Main/home/pi/DataBaseTable.db"
        # dbflag=True
    except sqlite3.OperationalError as Error:
        print("Database File Can't Be Found ")
        dbflag=False

    if dbflag:
        conn = sqlite3.connect(DATABASEPATH)
        conn.text_factory=str
    else:
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename(filetypes=[("Database", ".db")]) # show an "Open" dialog box and return the path to the selected file
        conn = sqlite3.connect(filename)
        print(filename)
    cursor = conn.cursor()
    check_query='SELECT Gate_ID,AcessSchemeID,Cat FROM EmployeeAccess WHERE Emp_ID=\''+EmpID+"\'"
    cursor.execute(check_query)
    EmpAccessRecord=cursor.fetchall()
    current_time = datetime.now()
    date = current_time.strftime("%d/%m/%Y")
    Time = current_time.strftime("%H:%M:%S")
    for rowx in EmpAccessRecord:
        GateID=rowx[0]
        AcessSchemaID=rowx[1]
        EmployeeCat=rowx[2]
        
        if str(GateID)==ThisGate:
            access_record=[ThisGate,EmpID,emp_name,date,Time]
            OpenDoorLock=True
            print("Employee",EmpID,"Have Access to Gate",str(GateID))
            cursor.execute(
                    '''INSERT INTO 
                    Access_Sheet(Gate_ID,Emp_ID,Emp_FirstName,Date,Time) 
                    VALUES (?,?,?,?,?,?);''', access_record)
            return OpenDoorLock
        else:
            OpenDoorLock=False
            print("Employee ",EmpID, "dont have  access to gate",str(GateID))

    conn.commit()
    conn.close()

# send and receive threads
# send_thread = thread.Thread(target=send,args=(sendSignal,))
recv_thread = thread.Thread(target=receive,args=(sendSignal,))
# starting threads
# send_thread.start()
recv_thread.start()