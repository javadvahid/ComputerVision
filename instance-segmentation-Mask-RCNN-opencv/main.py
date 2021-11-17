import cv2
import numpy as np
#load mask rcnn
net = cv2.dnn.readNetFromTensorflow("./frozen_inference_graph_coco.pb",


                                    "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt") 
#generate random colors for each class
colors = np.random.randint(0, 255, (80, 3))
img = cv2.imread('street.jpg')
img = cv2.resize(img, (500,500))
height, width, _ = img.shape

black_image = np.zeros((height, width, 3), np.uint8)
#detect objects
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
print(boxes.shape, masks.shape)
box = boxes[0, 0, 2]
print(box) 

for i in range(boxes.shape[2]):
    box = boxes[0, 0, i]
    class_id = box[1]
    score = box[2]
    if score<0.7:
      continue
    x = int(box[3]*width)
    y = int(box[4]*height)
    x2 = int(box[5]*width)
    y2 = int(box[6]*height)
    roi = black_image[y:y2, x:x2]
    roi_height, roi_width, _ = roi.shape
    mask = masks[i, int(class_id)]
    mask = cv2.resize(mask, (roi_width, roi_height))
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    print(mask)
    #cv2.imshow('mask', mask)
    cv2.rectangle(img, (x,y), (x2,y2), (255,0,0), 2)
    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    color = colors[int(class_id)]
    for cnt in contours:
        cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))	    
    	
cv2.imshow('Image', img)
cv2.imshow('segmentation', black_image)
cv2.waitKey(0)    