# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import matplotlib.pyplot as pt
import os
from PIL import Image
import csv
import pytesseract 
import re
import pandas as pd
from detect_faces import face_detection
from Image_enhancement import imageEnhancement
from converter import converter_to_specified_format
from crop_morphology import crop_address_image
# import tesseract


current_time = time.time()
# =================================================\
#    resizing code
def resize_toStandard(image, type):
	if type == "Adhar":
		or_ht, or_wd = [387, 533] 
	elif type == "old_pan":
		or_ht, or_wd = [633, 1012] 
	elif type == "new_pan":
		or_ht, or_wd = [501, 777]

	#### RESIZE ####
	te_ht = image.shape[0]
	te_wd = image.shape[1]
	
	ht_rt = ((or_ht - te_ht)/te_ht)*100
	wd_rt = ((or_wd - te_wd)/te_wd)*100
	
	
	width = int(image.shape[1] * (100+wd_rt)/100)
	height = int(image.shape[0] * (100+ht_rt)/100)
	dim = (width, height)
	return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# ===================================


# =========================================
def conditional_sort(element):
	return element[1]
# =========================================



# ==============================================
# Check wheater boxes are overlapping or not
def check_merge(box1, box2, i, j):
	x_, y_, _x, _y = box1
	a_, b_, _a, _b = box2

	w1 = abs(_x-x_)
	w2 = abs(_a-a_)
	h1 = abs(_x-x_)
	h2 = abs(_x-x_)
	h_thres = 10
	# print(i, j,box1, box2, (abs(x_-a_)+abs(_x-_a)<=w1+w2) , (abs(y_-b_)+abs(_y-_b)<=h1+h2), abs(y_-b_), abs(_y-_b),' h1:', h1, ' h2:', h2, ' w1:', w1, ' w2:', w2)
	if((abs(x_-a_)+abs(_x-_a)<=w1+w2) and (abs(y_-b_)+abs(_y-_b)<=h1+h2)):
		# print(abs(b_-y_) < h_thres, abs(_b-_y)<h_thres)
		if(abs(b_-y_) < h_thres and abs(_b-_y)<h_thres):
			# print('yes')
			return True 
	else:
		return False



# ==============================================


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())
	
# directory = args['image']



def do_for_all(original_image, index):
	# construct the argument parser and parse the arguments
	
	# load the input image and grab the image dimensions
	# image = cv2.imread(args["image"])
	# Reading Image
	if os.path.isfile(original_image): 
		image = cv2.imread(original_image)
	else:
		print('unable to read file in tesseract_text_detection.py')
		return
	# Resize Image to Standard Ratio
	image = resize_toStandard(image, "Adhar")

	temp = image
	orig = image.copy()
	(H, W) = image.shape[:2]
	
	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (args["width"], args["height"])
	rW = W / float(newW)
	rH = H / float(newH)
	
	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]
	
	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]
	
	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet("frozen_east_text_detection.pb")
	
	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()
	
	# show timing information on text prediction
	print("[INFO] text detection took {:.6f} seconds".format(end - start))
	
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
	
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
	
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
	
			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
	
			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
	
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
	
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	
	
	
	
	i=0
	# if not exist(os.path("./output")):
	# 	os.mkdir('./output')
	# loop over the bounding boxes
	
	temp = resize_toStandard(temp, "Adhar")
	# cv2.imshow('hey', temp)
	# print(boxes)
	
	mod_boxes = []
	for a,b,c,d in boxes:
		if(b<=290 and d<=290 and b>=56 and d>=56 and (c-a)>=50):
			mod_boxes.append([a,b,c,d])
	# print(mod_boxes)




	for i in range(len(boxes)):
		boxes[i][0] = boxes[i][0] - 5 # startX
		boxes[i][1] = boxes[i][1] - 5 # startY
		boxes[i][2] = boxes[i][2] + 12 # endX
		boxes[i][3] = boxes[i][3] + 5 # endY

	for i in range(len(boxes)):
		for j in range(i):
			box1 = boxes[i]
			box2 = boxes[j]
			if check_merge(box1, box2, i, j):
				# print("cool")
				# print("boxes : ", boxes,"\n box1 : =" ,box1,"\n box2 : ", box2)
				if boxes[i][0] < boxes[j][0]:
					boxes[i] = boxes[j] = [boxes[i][0], boxes[i][1], boxes[j][2], boxes[j][3]]
				else:
					boxes[i] = boxes[j] = [boxes[j][0], boxes[j][1], boxes[i][2], boxes[i][3]]
	for i in range(len(boxes)):
		for j in range(i):
			box1 = boxes[i]
			box2 = boxes[j]
			if check_merge(box1, box2, i, j):
				# print("cool")
				# print("boxes : ", boxes,"\n box1 : =" ,box1,"\n box2 : ", box2)
				if boxes[i][0] < boxes[j][0]:
					boxes[i] = boxes[j] = [boxes[i][0], boxes[i][1], boxes[j][2], boxes[j][3]]
				else:
					boxes[i] = boxes[j] = [boxes[j][0], boxes[j][1], boxes[i][2], boxes[i][3]]
	
	# for box in boxes:

	newboxes=[]
	flag = True
	#newboxes1=[]
	#print(boxes)
	for (a,b,c,d) in boxes:
		if(b<=290 and d<=290 and b>=56 and d>=56 and (c-a)>=70):
			if not len(newboxes):
				newboxes.append([a,b,c,d])
			else:
				for i in range(len(newboxes)):
					( sx, sy, ex, ey) = newboxes[i]
					#print(i, a, b, c, d, " : " ,sx, sy, ex, ey, (sx == a and sy == b), (ex == c and ey == d), ((sx == a and sy == b) and (ex == c and ey == d)))
					if ((sx == a and sy == b) and (ex == c and ey == d)):
						flag = False
						# print(newboxes)
				if flag:
					newboxes.append([a, b, c, d])
				flag = True


	#print(boxes)
	#print(newboxes)
	newboxes.sort(key = conditional_sort)
	# print(newboxes)
	newboxes.pop(0)

	#if len(newboxes) > 2:
		#newboxes[1][0] = newboxes[1][0] + (newboxes[1][2] - newboxes[1][1])*0.58
		#newboxes[2][0] = newboxes[2][0] + (newboxes[2][2] - newboxes[2][1])*0.71

	ind = -1
	# print(boxes, len(boxes))
	# print(mod_boxes)
	text_recognized = []
	# text_recognized.append(original_image.split('./')[len(original_image.split('./'))-1])
	print("############", index, "############")
	for (startX, startY, endX, endY) in newboxes:
		# scale the bounding box coordinates based on -the respective
		# ratios
		ind += 1
		if(ind==1):
			startX-= (startX-endX)*0.55
		elif(ind==2):
			startX-= (startX-endX)*0.45

		# print("i = ", ind , startX, startY, endX, endY)
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		imgName = "./output/crop"+str(index)+"_"+str(ind)+".png"
	
		# print("actual i: " , ind, "dim : ", startX, startY, endX, endY)
	
		cv2.imwrite(imgName, temp[startY: endY, startX: endX])

		text = pytesseract.image_to_string(Image.open(imgName), lang='eng', \
			config='--psm 8 --oem 3 -c tessedit_char_whitelist=  0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/')
		
		imgName = imgName.split('./')[len(imgName.split('./'))-1]
		imgName = imgName.split('/')[len(imgName.split('/'))-1]
		text_recognized.append(imgName)
		text_recognized.append(text)
		
		print("-=======",imgName,"======--")
		print(text)
		print("------------------")

		label = str(ind)
		# draw the bounding box on the image
		# cv2.rectangle(orig, (startX-10, startY-5), (endX+10, endY+5), (0, 255, ind*10), 2)
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, ind*10), 2)
		cv2.putText(orig,label,(startX-10, startY-5),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
	
	# show the output image
	print("===========================\n\n")
	
	if len(text_recognized) == 8:
		new_num = text_recognized.pop()
		n = ''
		for x in new_num:
			if (ord(x)<=57 and ord(x)>=48) or ord(x) == ord(' '):
				n+=x
		text_recognized.append(n) 
	p = "./fullCard/"+str(index)+".png"
	cv2.imwrite(p, orig)
	#pt.imshow(orig)
	#pt.show()
	# cv2.imshow("Text Detection", orig)
	# cv2.waitKey(0)
	
	return text_recognized

# files = os.listdir(directory)
if not os.path.exists("output"):
	os.makedirs('output')
if not os.path.exists("fullCard"):
	os.makedirs('fullCard')
if not os.path.exists('csv_file'):
	os.makedirs('csv_file')

k =0 
# for file in files:
# 	k +=1
# 	file = os.path.join(args['image'], file)
# 	do_for_all(file, k)
# 	print(time.time()-current_time)
# 	# break


if os.path.isfile('./csv_file/addhar_input.csv'):
	fileopen = pd.read_csv('./csv_file/addhar_input.csv')
	file_name = fileopen['aadhar_front_image'][0]

	# file_name = fileopen.readline().split('\n')[0]
	# file_name = './'+ file_name.split('./')[len(file_name.split('./'))-1]
	file_name = './adhar/'+file_name
	#print(file_name)
	if os.path.isfile(file_name):
		file_name = converter_to_specified_format(file_name, '.png')
		# print(file_name)
		csv_path = "./csv_file/adhar_card_output.csv"

		# This is the csv file to write the output  
		csv_file = open(csv_path, 'w')
		writer = csv.writer(csv_file)

		# writing titles in csv file 
		writer.writerow(['adhar_card_front','adhar_card_back', 'photo', 'name_image', 'name_recognized', 'dob_image', 'dob_recognized', 'gender_image', 'gender_recognized', 'adharNumber_image', 'adharNumber_recognized', 'address_recognised'])


		face_detection(file_name)
		imageEnhancement('./output/photo_out.png')
		first = []
		first.append(file_name.split('/')[len(file_name.split('/'))-1])
		text_recognized = do_for_all(file_name, k)
		

		# Reading address from adhar card
		print('[FETCHING ADDRESS ] --') 
		if os.path.isfile('./csv_file/addhar_input.csv'):
			address_file = fileopen['aadhar_back_image'][0]
			address_file = './adhar_back/'+ address_file
			# address_file = './adhar_back/'+ address_file.split('/')[len(address_file.split('/'))-1]

			if os.path.isfile(address_file):
				a = crop_address_image(address_file)
				print('=++++++++++++++++ Proper Address +++++++++++++++++')
				print(a[0])
				print('[FILE]:', a[1])
				print('------------------------------------')
				first.append(a[1].split('/')[len(a[1].split('/'))-1])
				text_recognized.append(a[0])
			else:
				print('[ADDRESS IMAGE NOT FOUND] -- ', address_file)
				
		else:
			print('aadhar_input.csv file is not in the directory')
		first.append('photo_out.png')

		for x in text_recognized:
			first.append(x)
		print(first)
		writer.writerow(first)
		csv_file.close()
	else:
		print('unable To Locate file name = ',file_name,'::in tesseract main')
	# fileopen.close()
else:
	print('unable To Locate file name = ','aadhar_input.csv','::in tesseract main')


# [[203,  48, 330,  77], [  9,  80, 155, 107], [  8,  12, 151,  44], [  9,  80, 155, 107], [  8,  12, 151,  44], [  7,  45,  65,  76], [268,  16, 333,  46], [ 10, 203, 108, 230], [  9,  80, 155, 107], [ 11, 165,  88, 192], [ 33, 189, 141, 215], [203,  48, 330,  77], [204,  14, 266,  45], [  8, 122, 135, 150], [ 67,  46, 153,  77], [  8, 122, 135, 150], [  8, 122, 135, 150], [  9, 280,  67, 304], [ 33, 189, 141, 215], [203,  48, 330,  77], [ 16, 251,  60, 278]]



# python text_detection4.py --image fullcard2 --east frozen_east_text_detection.pb 

# text = pytesseract.image_to_string(Image.open(filename), lang='eng', \
#         config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')-