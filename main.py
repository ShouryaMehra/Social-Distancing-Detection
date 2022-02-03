from scipy.spatial import distance as dist
import numpy as np
import cv2
import imutils
import io
from io import BytesIO
from PIL import Image, ImageDraw
from flask import Flask,jsonify,request,send_file
import json
import os
from dotenv import load_dotenv

# set-up bucket root
weightsPath = 'models/yolov3.weights'
configPath = 'models/yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
LABELS = open('models/coco.names').read().strip().split("\n")

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# set env for secret key
load_dotenv()

secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

def check_for_secret_id(request_data):    
    try:
        if 'secret_id' not in request_data.keys():
            return False, "Secret Key Not Found."
        
        else:
            if request_data['secret_id'] == secret_id:
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False,message

def detect_people(frame, net, ln, personIdx=0):
    (Height, Width) = frame.shape[:2]
    results = []
    Min_Confidence= 0.3
    NMS_Threshold= 0.3
    #Constructing a blob from the input frame and performing a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    centroids = []
    confidences = []
    
    #Looping over each of the layer outputs
    for output in layerOutputs:
    #Looping over each of the detections
        for detection in output:
            #Extracting the class ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #Filtering detections by:
            #1 Ensuring that the object detected was a person
            #2 Minimum confidence is met
            if classID == personIdx and confidence > Min_Confidence:
                #Scaling the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([Width, Height, Width, Height])
                (centerX, centerY, width, height) = box.astype("int")
                #Using the center (x, y)-coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                #Updating the list of bounding box coordinates, centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                
    #Applying non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, Min_Confidence, NMS_Threshold)
    #ensuring at least one detection exists
    if len(idxs) > 0:
        #Looping over the indexes we are keeping
        for i in idxs.flatten():
            #Extracting the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #Updating our results list to consist of the person prediction probability, bounding box coordinates and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results
@app.route('/social_distancing_detection',methods=['POST'])  #main function
def main():
    key = request.form['secret_id']
    request_data = {'secret_id' : key}
    secret_id_status,secret_id_message = check_for_secret_id(request_data)
    print ("Secret ID Check: ", secret_id_status,secret_id_message)
    if not secret_id_status:
        return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                        'success':False}) 
    else:
        img_params =request.files['image'].read()
        npimg = np.fromstring(img_params, np.uint8)
        #load image
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR) # load image
        
        Min_Confidence= 0.3
        NMS_Threshold= 0.3

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))

        violate = set()

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    #Checking if the distance between< number of pixels (60)
                    if D[i, j] < 60:
                        violate.add(i)
                        violate.add(j)

        #Looping over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            #Extract the bounding box and centroid coordinates, colour set to blue if okay
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (255, 0, 0)

            #Red if in violation
            if i in violate:
                color = (0, 0, 255)

            #Bounding box and centroid marking
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)  

        I = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(I.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)

        output = send_file(file_object, mimetype='image/PNG') 

    return output
if __name__ == '__main__':
    app.run()                       
                


