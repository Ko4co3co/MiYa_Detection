import sys
import glob
import os
import cv2
import numpy as np

img_dir = "./"
output_dir = "./"
for i in range(len(sys.argv)-1):
        if(sys.argv[i] == '-i'):
            img_dir = sys.argv[i+1]
        elif(sys.argv[i] == '-o'):
            output_dir = sys.argv[i+1]

print("IMG_DIR : {}".format(img_dir))
img_path = os.path.join(img_dir,'*.jpg')
img_glob = glob.glob(img_path)
print("Total Img : {}".format(len(img_glob)))
# 마우스 콜백 함수
annotation = []

def draw_box(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Down = {} {}".format(x,y))
        annotation.append(x)
        annotation.append(y)

    elif event == cv2.EVENT_LBUTTONUP:
        print("UP = {} {} ".format(x,y))
        annotation.append(x)
        annotation.append(y)
        print("Input Label :")
        label = input()
        if label != 'x':
            print(label)
            annotation.append(label)

def switching_Maximum(a,b,scale):
    maximum = max(a,b)
    minimum = min(a,b)
    maximum = maximum / scale
    minimum = minimum / scale
    return maximum, minimum

for i in range(len(img_glob)):
    annotation = []
    img = cv2.imread(img_glob[i])
    x_scale = img.shape[0]
    y_scale = img.shape[1]
    print(img_glob[i])
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_box)

    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    print(annotation)
    annotation[0] , annotation[2] = switching_Maximum(annotation[0] , annotation[2] ,x_scale)
    annotation[1] , annotation[3] = switching_Maximum(annotation[1] , annotation[3] ,y_scale)

    filename =str(i) + ".txt"
    filename = os.path.join(output_dir,filename)
    fileptr = open(filename,'w')
    # xmin ymin xmax ymax , label
    for k in range(len(annotation)):
        fileptr.write(str(annotation[k]))
        fileptr.write('\n')
        
    fileptr.close()

