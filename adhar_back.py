import image_slicer
import math
from PIL import Image
import pytesseract
import os
from converter import converter_to_specified_format
import csv
from difflib import SequenceMatcher


def address_crop(file_path):
    image= file_path
    print('address ', image)
    image = converter_to_specified_format(image, '.png')
    tiles = image_slicer.slice(image, 4, save=False)
    image_slicer.save_tiles(tiles, directory='./output',\
                                prefix='slice', format='png')
                            

    arr=[]
    for i in range (0,4):
        if not os.path.exists('./output'):
            os.makedirs('./output')


        if(i==0):
            p='./output/slice_01_01.png'
        elif(i==1):
            p='./output/slice_01_02.png'
        elif(i==2):
            p='./output/slice_02_01.png'
        elif(i==3):
            p='./output/slice_02_02.png'

        text = pytesseract.image_to_string(Image.open(p), lang='eng', \
                                       config='--psm 3 --oem 3 -c tessedit_char_whitelist= 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/')
        

        if(i==0):
            t1=text
            arr.append(len(text))
        elif(i==1):
            t2=text
            arr.append(len(text))
        elif(i==2):
            t3=text
            arr.append(len(text))
        elif(i==3):
            t4=text
            arr.append(len(text))

    import heapq
    df = heapq.nlargest(4, range(len(arr)), key=arr.__getitem__)

    original = Image.open(image)
    w, h = original.size
    if(len(t2)>len(t4) and len(t2)>len(t1)):
        print("text is on right")
        l=0.47*w
        r=0.01*h
        v=w
        s=0.99*h
    elif(len(t2)>len(t1) and len(t4)>len(t3)):
        print("text is on right")
        l=0.47*w
        r=0.01*h
        v=w
        s=0.99*h
    elif(len(t1)>len(t2) and len(t1)>len(t3)):
        print("text is on left")
        l=0.001*w
        r=0.01*h
        v=0.5*w
        s=0.99*h
    elif(len(t1)>len(t2) and len(t3)>len(t4) and (df[1]==2 or df[1]==0)):
        print("text is on left")
        l=0.001*w
        r=0.01*h
        v=0.5*w
        s=0.99*h
    elif(len(t3)>len(t1) and len(t4)>len(t2) and (df[1]==2 or df[1]==3)):
        print("text is on bottom")
        l=0.001*w
        r=0.35*h
        v=w
        s=0.99*h
    elif(len(t3)+len(t4) > len(t1)+len(t2) and (df[1]==2 or df[1]==3)):
        print("text is on bottom")
        l=0.001*w
        r=0.35*h
        v=w
        s=0.99*h
    else:
        print("text is on top")
        l=0.001*w
        r=0.001*h
        v=w
        s=0.60*h

    cropped_image= original.crop((l, r, v, s)).save('output/cropped_.png')
    text = pytesseract.image_to_string(Image.open('output/cropped_.png'), lang='eng', \
                                       config='--psm 3 --oem 3 -c tessedit_char_whitelist= 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/')
    print('----------------text detected on address-------------------')
    print(text)
    print('================================')
    keyword1=[]
    keyword3=[]
    after_keyword=[]    
    keyword = 'Address:'
    before_keyword, keyword, after_keyword = text.partition(keyword)

    # string matching 
    if (len(after_keyword) == 0):
        for x in text.split('\n'):
            if len(x)>=6 and SequenceMatcher(None, x, 'Address:').ratio()>0.75:
                keyword2=x
                z, x, after_keyword = text.partition(keyword2)
    # before_keyword, keyword, after_keyword = text.partition(keyword)
    
    for x in after_keyword.split(' '):
        if len(x)==6 and x.isdigit():
            keyword1=x
    print(after_keyword)
    if (len(keyword1) == 0):
        keyword1 = '\n\n'
    a,b,c = after_keyword.partition(keyword1)
    address = (a + b)
    if (len(address) <= 6):
        keyword3 = '\n\n\n'
        a,b,c = c.partition(keyword3)
        address = (a)
    # print(address)
    # print(address)
    return address

