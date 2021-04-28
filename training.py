#xWalk Training Model 
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from glob import glob
import sys
from PIL import Image
import cv2 as cv
import time 

starttime = time.time()

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
model_path = "./"
PATH_TO_CKPT = model_path + MODEL_NAME + '/frozen_inference_graph.pb'

# def download_model():
#     import six.moves.urllib as urllib
#     import tarfile

#     opener = urllib.request.URLopener()
#     opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#     tar_file = tarfile.open(MODEL_FILE)
#     for file in tar_file.getmembers():
#         file_name = os.path.basename(file.name)
#         if 'frozen_inference_graph.pb' in file_name:
#             tar_file.extract(file, os.getcwd())

def load_graph():
    #if not os.path.exists(PATH_TO_CKPT):
    #    download_model()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(fid.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
    return graph


def select_boxes(boxes, classes, scores, score_threshold=0, target_class=10):
    #param boxes:
    #param classes:
    #param scores:
    #param target_class: default traffic light id in COCO dataset is 10
   
    sq_scores = np.squeeze(scores)
    sq_classes = np.squeeze(classes)
    sq_boxes = np.squeeze(boxes)

    sel_id = np.logical_and(sq_classes == target_class, sq_scores > score_threshold)

    return sq_boxes[sel_id]

class TLClassifier(object):
    def __init__(self):

        self.detection_graph = load_graph()
        self.extract_graph_components()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        #running the first session to "warm up"
        dummy_image = np.zeros((100, 100, 3))
        self.detect_multi_object(dummy_image,0.1)
        self.traffic_light_box = None
        self.classified_index = 0

    def extract_graph_components(self):
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_multi_object(self, image_np, score_threshold):
        """
        Return detection boxes in a image

        :param image_np:
        :param score_threshold:
        :return:
        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        sel_boxes = select_boxes(boxes=boxes, classes=classes, scores=scores,
                                 score_threshold=score_threshold, target_class=10)

        return sel_boxes

def crop_roi_image(image_np, sel_box):
    im_height, im_width, _ = image_np.shape
    (left, right, top, bottom) = (sel_box[1] * im_width, sel_box[3] * im_width,
                                  sel_box[0] * im_height, sel_box[2] * im_height)
    cropped_image = image_np[int(top):int(bottom), int(left):int(right), :]
    return cropped_image
        
test_file = "JPG/IMG_1594.jpg"

im = Image.open(test_file)
image_np = np.asarray(im)
tlc=TLClassifier()

boxes=tlc.detect_multi_object(image_np,score_threshold=0.2)

cropworked = True
try:  
    cropped_image=crop_roi_image(image_np,boxes[0])
except: 
    cropworked = False
    
if cropworked:
    plt.imshow(cropped_image)
    plt.show()
    plt.savefig(r'trafficlight.jpg')
else: 
    print("Traffic light not found")

####
####IDENTIFYING THE COLOR OF THE TRAFFIC LIGHT 
####

def redFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    # Range for lower red
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])
    mask_red1 = cv.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([170,120,70])
    red_upper = np.array([180,255,255])
    mask_red2 = cv.inRange(hsv, red_lower, red_upper)

    mask_red = mask_red1 + mask_red2

    red_output = cv.bitwise_and(image, image, mask=mask_red)
    return red_output

def yellowFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    # Range for Yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

    yellow_output = cv.bitwise_and(image, image, mask=mask_yellow)

    return yellow_output

def greenFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    # cv.imshow("green hsv image", hsv)
    # cv.waitKey(0)

    #Range for Green
    green_lower = np.array([36, 0, 0])
    green_upper = np.array([85, 255,255])
    mask_green = cv.inRange(hsv, green_lower, green_upper)

    green_output = cv.bitwise_and(image, image, mask=mask_green)

    return green_output

def rygFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    # Range for lower red (red is split on hsv scale)
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])
    mask_red1 = cv.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([170,120,70])
    red_upper = np.array([180,255,255])
    mask_red2 = cv.inRange(hsv, red_lower, red_upper)

    # Range for Yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

    #Range for Green
    green_lower = np.array([36, 0, 0])
    green_upper = np.array([85, 255,255])
    mask_green = cv.inRange(hsv, green_lower, green_upper)

    combined_mask = mask_red1 + mask_red2 + mask_yellow + mask_green

    ryg_output = cv.bitwise_and(image, image, mask=combined_mask)
    return ryg_output

def rygwFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    # Range for lower red (red is split on hsv scale)
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])
    mask_red1 = cv.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([170,120,70])
    red_upper = np.array([180,255,255])
    mask_red2 = cv.inRange(hsv, red_lower, red_upper)

    # Range for Yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

    #Range for Green
    green_lower = np.array([36, 0, 0])
    green_upper = np.array([85, 255,255])
    mask_green = cv.inRange(hsv, green_lower, green_upper)

    #Range for White
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)
    mask_white = cv.inRange(hsv, lower_white, upper_white)

    combined_mask = mask_red1 + mask_red2 + mask_yellow + mask_green + mask_white

    rygw_output = cv.bitwise_and(image, image, mask=combined_mask)
    return rygw_output

# buggy output -> not quite an image...
def histEqual(image):
    # img = cv.imread('fingerptint.jpg', cv.IMREAD_UNCHANGED)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(image)
    cv.imshow("equalized image", dst)
    cv.waitKey(0)
    return dst

# INTENSITY THRESHOLDING ALGORITHM HERE
def intensityThresh(gray, threshold = 0):

    ret, thresh1 = cv.threshold(gray,threshold,255,cv.THRESH_TOZERO)
    return thresh1

def main(color="red"):
    filename = "trafficlight.jpg"
    
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    ## [Resize Image]
    #want the height to be H = 200 pixels
    H = 300
    scale = src.shape[0] / H
    resized = cv.resize(src, (int(src.shape[1]/scale), int(src.shape[0]/scale)))
    ## [Resize Image]

    ## [Red Filter]
    reddened = redFilter(resized)
    ## [Red Filter]

    ## [Yellow Filter]
    yellowed = yellowFilter(resized)
    ## [Yellow Filter]

    ## [Green Filter]
    greened = greenFilter(resized)
    ## [Green Filter]

    ## [Red, Yellow and Green Filter]
    ryg = rygFilter(resized)
    ## [Red, Yellow, and Green Filter]

    # ## [Red, Yellow and Green + White Filter]
    # rygw = rygwFilter(resized)
    # ## [Red, Yellow, and Green + White Filter]

    if (color == "red"):
        filtered = reddened
    elif (color == "yellow"):
        filtered = yellowed
    elif (color == "green"):
        filtered = greened
    else:
        filtered = ryg


    gray = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    gray = intensityThresh(gray, threshold = 0)
  
    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    gray = cv.medianBlur(gray, 15)
    ## [reduce_noise]
    
    ## [houghcircles]
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=20,
                               minRadius=(int(H/100)), maxRadius = 50)
    ## [houghcircles]

    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(resized, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(resized, center, radius, (255, 0, 255), 3)

    return circles


if __name__ == "__main__":
    if cropworked == True: 
        r_circles = main(color="red")
        y_circles = main(color="yellow")
        g_circles = main(color="green")
        num_circles = 0
        color_out = "none"
        if g_circles is not None and len(g_circles) > num_circles:
            color_out = "green"
        if y_circles is not None and len(y_circles) >= num_circles:
            color_out = "yellow"
        if r_circles is not None and len(r_circles) >= num_circles:
            color_out = "red"
        sys.stdout.write(color_out)


endtime = time.time()
print("\n")
print("time: ",str(endtime-starttime))

