import pyodbc
from PIL import Image
import io

# Establish a connection to the database
connection_string = 'DRIVER={SQL Server};SERVER=DESKTOP-63B15F9\\MSSQLSERVER01;DATABASE=SaafCity_Database_2;UID=sa;PWD=1234'
connection = pyodbc.connect(connection_string)

# Create a cursor to execute SQL queries
cursor = connection.cursor()

# Execute the query to retrieve the image data
query = "SELECT Complaint_Image FROM Complaints WHERE Complaint_ID = ?"
image_id = 4061  # Change this to the desired Complaint_ID
cursor.execute(query, (image_id,))  # Pass the parameter as a tuple

# Retrieve the image data from the query result
result = cursor.fetchone()
image_data = result[0]

# Convert the image data to a PIL.Image object
image = Image.open(io.BytesIO(image_data))

# Display the image
image.show()

# Close the cursor and connection
cursor.close()
connection.close()
