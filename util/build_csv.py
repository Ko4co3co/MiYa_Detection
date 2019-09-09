import pandas as pd 
import os

class csv():
    def __init__(self,label_dir='./label',image_dir='./img',csv_output='./output.csv'):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.csv_output = csv_output

    def file_list(self):
        label_list = os.listdir(self.label_dir)
        image_list = os.listdir(self.image_dir)
        return label_list, image_list

    #xmin , ymin, xmax, ymax, label
    def label_read(self,label_file):
        label_name = os.path.join(self.label_dir,label_file)
        fileptr = open(label_name,'r')
        label = []
        r = fileptr.readlines()
        for i in r:
            label.append(i.replace('\n',''))
        
        fileptr.close()
        return label

    def make_csv(self):
        annotation = []
        label, image = self.file_list()
        print(label)
        print(image)
        for i in range(len(label)):
            t = self.label_read(label[i])
            xmax = t[0]
            ymax = t[1]
            xmin = t[2]
            ymin = t[3]
            category = t[4]
            filename = image[i]
            annotation.append([filename,xmax,ymax,xmin,ymin,category])

        df = pd.DataFrame(annotation,columns=['filename','xmax','ymax','xmin','ymin','category'])
        df.to_csv(self.csv_output,index=False)
        print("Success Make {} File".format(self.csv_output))

label_dir = './label'
image_dir = './img'
csv_output='./train.csv'
build=csv(label_dir,image_dir,csv_output)
build.make_csv()
