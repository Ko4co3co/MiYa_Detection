import numpy as np 
import glob 
import tensorflow as tf 
import cv2 
import label_map_util 


def inference(img,graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict,feed_dict={image_tensor : img})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
        return output_dict

def visualization_img(img,label_map,output,score_threshold=0.5):
    width , height = img.shape[1:3]
    img = np.squeeze(img,axis=0)
    #Load inference Result
    scores = output['detection_scores']
    bboxes = output['detection_boxes'] * [width,height,width,height]
    classes = output['detection_classes']
    draw_boxes = []
    for i in range(len(scores)):
        if scores[i] >= score_threshold:
            draw_boxes.append([bboxes[i],classes[i],scores[i]])
    if len(draw_boxes) > 0:
        for k in range(len(draw_boxes)):
            ymin = int(draw_boxes[k][0][0])
            xmin = int(draw_boxes[k][0][1])
            ymax = int(draw_boxes[k][0][2])
            xmax = int(draw_boxes[k][0][3])
            img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color=(0,255,0),thickness =3)
            cv2.putText(img,label_map[draw_boxes[k][1]]['name'],(xmin,ymin-10),cv2.FONT_HERSHEY_PLAIN,0.7, (100,255,100),1)           

    return img

# MODEL lOAD
MODEL_DIR = '../data'
PATH_TO_FROZEN_GRAPH = MODEL_DIR+ '/output_inference_graph.pb'
PATH_TO_LABELS = MODEL_DIR + '/model.pbtxt'
SAVE_DIR = '../eval'
NUM_CLASSES = 1
# Graph Load 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#Category Load
label_map = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Image Load
PATH_TO_TEST_IMAGE_PATH = '../img'
IMG_GLOB_PATH = PATH_TO_TEST_IMAGE_PATH + '/*'
TEST_IMAGE_LIST = glob.glob(IMG_GLOB_PATH)
OUTPUT_IMAGE_DIR = '../output/image/'
print(TEST_IMAGE_LIST[0])
for i in range(len(TEST_IMAGE_LIST)):
#for i in range(0,1):
    test = cv2.imread(TEST_IMAGE_LIST[i],cv2.IMREAD_COLOR)
    # (N , ? , ? , 3) = Input Image , Width , Height , Channel 
    # Expand Dims으로 1번째 차원 확장으로 Rank를 맞춘다.
    test = np.expand_dims(test,axis=0)
    output = inference(test,detection_graph)
    #print(output)
    test = visualization_img(test,label_map,output)
    OUTPUT_FILE_PATH = OUTPUT_IMAGE_DIR + 'test ' + str(i) + '.jpg'
    cv2.imwrite(OUTPUT_FILE_PATH,test)



