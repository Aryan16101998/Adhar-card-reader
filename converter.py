from PIL import Image
import os
from pdf2image import convert_from_path

def converter_to_specified_format(imageName, format):
	print(imageName)
	# print(imageName, imageName.split('.')[len(imageName.split('.'))-1])
	
	try:
		if imageName.split('.')[len(imageName.split('.'))-1] == format:
			print('convertToPng Return NULL imageName', imageName)
			return imageName

		if(os.path.splitext(imageName)[1]== '.pdf'):
			imageName = convert_from_path(imageName)
			i=0
			for image in imageName:
				image.save('./output/converterd_image'+str(i)+format, 'PNG')
				i+=1
				return './output/converterd_image'+str(i)+format
		else:
			fileName=os.path.splitext(imageName)[0]
			im = Image.open(imageName)
			if format == '.jpeg':
				print('converted')
				im = im.convert('RGB')
			fileName = fileName.split('/')[len(fileName.split('/'))-1].split('.')
			fileName = os.path.join("./output/",fileName[len(fileName)-1])
			print("converted Image == >",fileName + format)
			im.save(fileName + format)
			return fileName + format
	except FileNotFoundError:
		print('unable to read file')
		return ' '
