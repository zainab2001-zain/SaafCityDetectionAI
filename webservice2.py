import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from models.research.object_detection.utils import label_map_util
import pyodbc
import base64
import io
import time

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = 'D:/fyp/Final Fyp/detection model/New folder/training_demo/exported_models/final/saved_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = 'D:/fyp/Final Fyp/detection model/New folder/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.50

# Load the saved model
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

# Load label map data
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Establish a connection to the SQL Server database
connection_string = 'DRIVER={SQL Server};SERVER=DESKTOP-63B15F9\MSSQLSERVER01;DATABASE=SaafCity_Database_2;UID=sa;PWD=1234'

# Define the delay between iterations (in seconds)
DELAY_SECONDS = 2

EXPECTED_HEIGHT = 300  # Desired height of the input image
EXPECTED_WIDTH = 300  # Desired width of the input image

while True:
    # Establish a connection to the SQL Server database
    connection = pyodbc.connect(connection_string)

    # Create a cursor to execute queries
    cursor = connection.cursor()

    # Fetch all the complaint IDs, complaint images, and verification images from the 'complaints' table where department ID is null
    query = "SELECT Complaint_ID, Complaint_Image, Verification_Image FROM Complaints WHERE Depart_ID IS NULL"
    cursor.execute(query)
    rows = cursor.fetchall()

    for row in rows:
        complaint_id = row.Complaint_ID
        complaint_image_base64 = row.Complaint_Image
        verification_image_base64 = row.Verification_Image

        # Convert the base64 image to PIL Image
        image_data = row[0]
        complaint_image = Image.open(io.BytesIO(image_data))
        complaint_image.show()

        # Save the PIL Image to a temporary file
        complaint_image_path = 'D:/fyp/web application/webservice/temp_complaint_image.jpg'
        complaint_image.save(complaint_image_path)

        # Perform detection on the complaint image
        complaint_image_np = cv2.imread(complaint_image_path)
        complaint_image_rgb = cv2.cvtColor(complaint_image_np, cv2.COLOR_BGR2RGB)
        complaint_image_resized = cv2.resize(complaint_image_rgb, (EXPECTED_WIDTH, EXPECTED_HEIGHT))
        complaint_image_expanded = np.expand_dims(complaint_image_resized, axis=0).astype(np.uint8)

        complaint_input_tensor = tf.convert_to_tensor(complaint_image_expanded)

        complaint_detections = detect_fn(complaint_input_tensor)

        complaint_num_detections = int(complaint_detections.pop('num_detections'))
        complaint_detections = {key: value[0, :complaint_num_detections].numpy()
                                for key, value in complaint_detections.items()}
        complaint_detections['num_detections'] = complaint_num_detections

        complaint_detection_labels = []
        for i in range(complaint_num_detections):
            class_id = int(complaint_detections['detection_classes'][i])
            class_name = category_index[class_id]['name']
            confidence = float(complaint_detections['detection_scores'][i])
            if confidence >= MIN_CONF_THRESH:
                complaint_detection_labels.append(class_name)

        department_id = None
        if 'garbage' in complaint_detection_labels:
            department_id = 1
        elif 'sewage' in complaint_detection_labels:
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

    if len(rows) == 0:
        print("No complaints found with department ID as null or complaint status as 'Rejected'.")

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Wait for the specified delay before the next iteration
    time.sleep(DELAY_SECONDS)
