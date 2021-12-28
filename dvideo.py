import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app
import random
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import colorsys
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2
import numpy as np
import pytesseract
import cv2
import glob 
#change path untill test-env eg 'C:/xxxxxxxxxxxx/venv/Lib/tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = 'C:/Users/Mahesh/Desktop/digital_image_processing/test-env/venv/Lib/tesseract.exe'  # your path may be different

#extracts co-ordinates of the bounding box
def format_boxes(bboxes, image_height, image_width):
	for box in bboxes:
		ymin = int(box[0] * image_height)
		xmin = int(box[1] * image_width)
		ymax = int(box[2] * image_height)
		xmax = int(box[3] * image_width)
		box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
	return bboxes



#Reads the object class name 
def read_class_names(class_file_name):
	names = {}
	with open(class_file_name, 'r') as data:
		for ID, name in enumerate(data):
			names[ID] = name.strip('\n')
	return names

    
def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES=[8,16,32]
    ANCHORS=[12,16,19,36,40,28,36,75,76,55, 72,146, 142,110, 192,243, 459,401]
    NUM_CLASS=1
    XYSCALE=[1.2, 1.1, 1.05]
    input_size = 416
    ###################################################################################################################
    output='./detections/results11.mp4'
    CLASSES="./data/classes/obj.names"
    video_path = './data/video/cars1.mp4'
    basepath=r'detections/frames/*.png'

    # extracts video name 
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # starts video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # VideoCapture returns float values by default
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, codec, fps, (width, height))

    frame_num = 0
    #draws boundary boxes around the detected image
    def draw_bbox(image, bboxes, info = False, counted_classes = None, show_label=True, allowed_classes=list(read_class_names(CLASSES)), read_plate = False):
        classes = read_class_names(CLASSES)
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes):
            if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
            coor = out_boxes[i]
            fontScale = 0.5
            score = out_scores[i]
            class_ind = int(out_classes[i])
            class_name = classes[class_ind]
            if class_name not in allowed_classes:
                continue
            else:

                bbox_color = colors[class_ind]
                bbox_thick = int(0.6 * (image_h + image_w) / 600)
                c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

               
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #fills colour

                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

                if counted_classes != None:
                    height_ratio = int(image_h / 25)
                    offset = 15
                    for key, value in counted_classes.items():
                        cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                        offset += height_ratio
        return image


    def crop_objects(img, data, path, allowed_classes,frames_num):
            
            boxes, scores, classes, num_objects = data
            class_names = read_class_names(CLASSES)
            #create a dictionary to hold count of objects 
            counts = dict()
            for i in range(num_objects):
                # gets the count of class objects for image name
                class_index = int(classes[i])
                class_name = class_names[class_index]
                if class_name in allowed_classes:
                    counts[class_name] = counts.get(class_name, 0) + 1
                    # get boundary box coords
                    xmin, ymin, xmax, ymax = boxes[i]
                    # crops detection from image ( 5 pixels padding around all edges)
                    cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
                    # creates image name and joins it to the path
                    img_name = class_name + '_' + str(counts[class_name]) +'_'+  str(frame_num) + '.png'
                    img_path = os.path.join(path, img_name )
                    # saves image
                    cv2.imwrite(img_path, cropped_img)
                else:
                    continue
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        
        
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.5
        )
		
		
		
        # formats bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
		
        class_names = read_class_names(CLASSES)

        # by default allows all classes in .names file
        allowed_classes = list(class_names.values())
		
	    
        crop_rate = 50 # captures image every 50 many frames
        crop_path = os.path.join(os.getcwd(), 'detections')
        try:
            os.mkdir(crop_path)
        except FileExistsError:
            pass
        if frame_num % crop_rate == 0:
            final_path = os.path.join(crop_path, 'frames')
            try:
                os.mkdir(final_path)
            except FileExistsError:
                pass          
            crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes,frame_num)
        else:
            pass


        image = draw_bbox(frame, pred_bbox, info=False, allowed_classes=allowed_classes, read_plate=False)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("result", result)
        
        out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


    cv2.destroyAllWindows()
    x = glob.glob(basepath)
    for i in range(len(x)):
        img = cv2.imread(x[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, img1 = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        predicted_result = pytesseract.image_to_string(img1, lang ='eng',config ='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqurstuvwxyz')
        filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
        print("Image ",i," = ",filter_predicted_result)

    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass