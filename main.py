import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from models.research.object_detection.utils import label_map_util
import pyodbc
import base64
import io

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = 'D:/fyp/Final Fyp/detection model/New folder/training_demo/exported_models/final/saved_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = 'D:/fyp/Final Fyp/detection model/New folder/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.40

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

# Fetch all the complaint IDs and complaint images from the 'complaints' table where department ID is null
query = "SELECT Complaint_ID, Complaint_Image FROM Complaints WHERE Depart_ID IS NULL"

cursor.execute(query)
rows = cursor.fetchall()

for row in rows:
    complaint_id = row.Complaint_ID
    complaint_image_base64 = row.Complaint_Image

    # Convert the base64 image to PIL Image
    image_data = row[1]
    complaint_image = Image.open(io.BytesIO(image_data))
    complaint_image.show()
    # Save the PIL Image to a temporary file
    complaint_image_path = 'D:/fyp/web application/webservice/temp_image.jpg'
    complaint_image.save(complaint_image_path)

    # Get the detection labels for the image
    labels = get_detection_labels(complaint_image_path, MIN_CONF_THRESH)

    department_id = None
    if 'garbage' in labels:
        department_id = 1
    elif 'sewage' in labels:
        department_id = 2

    if department_id is not None:
        cursor.execute("UPDATE Complaints SET Depart_ID = ? WHERE Complaint_ID = ?", department_id, complaint_id)
        connection.commit()
        print("Department ID updated for Complaint ID:", complaint_id)
        # Retrieve the updated complaint record from the database
        cursor.execute("SELECT * FROM Complaints WHERE Complaint_ID = ?", complaint_id)
        updated_complaint = cursor.fetchone()

        # Print the updated complaint details
        print("Updated Complaint Details:")
        print("Complaint ID:", updated_complaint.Complaint_ID)
        print("Complainant Email:", updated_complaint.Complainant_Email)
        print("Department ID:", updated_complaint.Depart_ID)
    else:
        print("No valid department ID found based on the detected labels.")

if len(rows) == 0:
    print("No complaints found with department ID as null.")

# Close the cursor and connection
cursor.close()
connection.close()
