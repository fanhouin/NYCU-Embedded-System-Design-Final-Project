# import libraries
import os
import cv2
import imutils
import time
import pickle
import numpy as np
import socket
from imutils.video import FPS
from imutils.video import VideoStream
import threading
import dlib

def renderFace(im, landmarks, color=(0, 255, 0), radius=3):
	for p in landmarks.parts():
		cv2.circle(im, (p.x, p.y), radius, color, -1)

# Dlib facial landmarks model的path
predictor_path = "models/shape_predictor_5_face_landmarks.dat"

# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("output/recognizer", "rb").read())
le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
# print("Starting Video Stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
ip="0.0.0.0"
port=6666
s=socket.socket(socket.AF_INET , socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((ip,port))
s.listen(5)
conn, addr = s.accept()
t = 0
detector2 = dlib.get_frontal_face_detector()
predictor2 = dlib.shape_predictor(predictor_path)

# start the FPS throughput estimator
fps = FPS().start()
# loop over frames from the video file stream
def recvlen(sock, count):
	buf = ''
	while count:
		newbuf = sock.recv(count)
		if not newbuf: 
			return None
		for i in newbuf.decode('utf-8'):
			if i == '\0':
				return buf
			else:
				buf += i
			count -= 1
	return buf

def recvimg(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def send_img(conn, imglen, stringData):
	conn.send(imglen)
	conn.send(stringData)
	
while True:
	length = recvlen(conn, 10)
	data = recvimg(conn, int(length))
	# data=conn.recv(1000000)
	
	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frame = np.fromstring(data, np.uint8)
	frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	dets = detector2(frame, 1)

	for k, d in enumerate(dets):
		shape = predictor2(frame, d)
		renderFace(frame, shape)

	frame = cv2.resize(frame, (160, 120)) 
	ret, buffer = cv2.imencode(".png",frame)
	data = np.array(buffer)
	
	
	stringData = data.tostring()
	# print(len(stringData))
	
	imglen = str(len(stringData))
	while len(imglen) < 10:
		imglen += '\0'
	if t != 0:
		t.join()
	
	t = threading.Thread(target=send_img,args=(conn, imglen.encode('utf-8'), stringData))
	t.start()

	# conn.send(imglen.encode('utf-8'))
	# conn.send(stringData)
	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
cv2.destroyAllWindows()
