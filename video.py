import numpy as np
import argparse
import cv2
import time
import os 
import csv

def video_capture(file_name):
	fileopen = open('./csv_file/video_inout.csv', 'w')
	writer = csv.writer(fileopen)
	out_array =[]
	folder_name = './video_frame'
	vidcap = cv2.VideoCapture(file_name)
	count = 0
	success = True
	fps = int(vidcap.get(cv2.CAP_PROP_FPS))
	frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration = frame_count/fps
	z=duration/5
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	while success:
		success,image = vidcap.read()
		if count%(int(z)*fps) == 0 :
			cv2.imwrite(folder_name+'/frame'+str(count)+'.png',image)
			current_time = time.time()
			net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
			image = cv2.imread(folder_name+'/frame%d.png'%count)
			(h, w) = image.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
			net.setInput(blob)
			detections = net.forward()
			for i in range(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > 0.5:
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")
					text = "{:.2f}%".format(confidence * 100)
					y = startY - 10 if startY - 10 > 10 else startY + 10
					cv2.imwrite(folder_name+'/frame_'+str(count)+'.png', image[startY-20: endY+20, startX-20: endX+20])
					arr = folder_name+'/frame_'+str(count)+'.png'
					arr =  arr.split('/')[len(arr.split('/'))-1]	
					out_array.append(arr)
					print(time.time()-current_time)
		count+=1
	writer.writerow(out_array)
	fileopen.close()
	return True

if os.path.isfile('./csv_file/video_inout.csv'):
	fileopen = open('./csv_file/video_inout.csv')

	file_name = fileopen.readline().split('\n')[0]
	file_name = './video_frame/'+ file_name.split('/')[len(file_name.split('/'))-1]
	if os.path.isfile(file_name):
		file_name = video_capture(file_name)
	else: 
		print('unable to find test_file.mp4 in video.py')