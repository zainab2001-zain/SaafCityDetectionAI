import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from models.research.object_detection.utils import label_map_util
import pyodbc
import io

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = 'D:/fyp/Final Fyp/detection model/New folder/training_demo/exported_models/final/saved_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = 'D:/fyp/Final Fyp/detection model/New folder/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.60

# Load the saved model
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

# Load label map data
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Establish a connection to the SQL Server database
connection_string = 'DRIVER={SQL Server};SERVER=DESKTOP-63B15F9\MSSQLSERVER01;DATABASE=SaafCity_Database_2;UID=sa;PWD=1234'
connection = pyodbc.connect(connection_string)

# Create a cursor to execute queries
cursor = connection.cursor()

def get_detection_labels(image_path, min_score_thresh):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    detection_labels = []
    for i in range(num_detections):
        class_id = detections['detection_classes'][i]
        class_name = category_index[class_id]['name']
        confidence = detections['detection_scores'][i]
        if confidence >= min_score_thresh:
            detection_labels.append(class_name)

    return detection_labels

# Provide the path to your image
image_path = 'D:/fyp/Final Fyp/detection model/object detection/images/garbage1.jpg'

# Get the detection labels for the image
labels = get_detection_labels(image_path, MIN_CONF_THRESH)

# Convert the detected labels to a string
labels_string = ', '.join(labels)

# Print the detected labels
print("Detected Labels:", labels_string)

# Determine the department ID based on the detected labels
department_id = None
if 'garbage' in labels:
    department_id = 1
elif 'sewage' in labels:
    department_id = 2

# Update the department ID in the 'complaints' table
if department_id is not None:
    # Fetch the complaint ID from the 'complaints' table where department ID is null
    cursor.execute("SELECT TOP 1 Complaint_ID FROM omplaints WHERE department_id IS NULL")
    row = cursor.fetchone()

    if row is not None:
        complaint_id = row.complaint_id
        cursor.execute("UPDATE complaints SET department_id = ? WHERE complaint_id = ?", department_id, complaint_id)
        connection.commit()
        print("Department ID updated for Complaint ID:", complaint_id)
    else:
        print("No complaint found with department ID as null.")
else:
     print("No valid department ID found based on the detected labels.")
