import cv2  # Import the OpenCV library for computer vision tasks

# Path to the Haar Cascade XML file for detecting number plates
harcascade = "model/haarcascade_russian_plate_number.xml"

# Initialize video capture from the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

# Set the width of the video frame
cap.set(3, 640)  # Width of the frame in pixels

# Set the height of the video frame
cap.set(4, 480)  # Height of the frame in pixels

# Minimum area (in pixels) for a detected object to be considered a valid number plate
min_area = 500

# Initialize a counter for saved images
count = 0

# Start an infinite loop to continuously capture frames from the webcam
while True:
    # Capture a single frame from the webcam
    success, img = cap.read()

    # Load the Haar Cascade classifier for number plate detection
    plate_cascade = cv2.CascadeClassifier(harcascade)
    
    # Convert the captured frame to grayscale (required for Haar Cascade detection)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect number plates in the grayscale image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Iterate through each detected number plate
    for (x, y, w, h) in plates:
        # Calculate the area of the detected number plate
        area = w * h

        # Check if the area of the detected plate is larger than the minimum area threshold
        if area > min_area:
            # Draw a green rectangle around the detected number plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add a label above the rectangle indicating it’s a number plate
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract the region of interest (ROI) where the number plate is located
            img_roi = img[y: y + h, x: x + w]
            # Display the ROI in a separate window
            cv2.imshow("ROI", img_roi)

    # Display the frame with detected number plates and annotations
    cv2.imshow("Result", img)

    # Check if the 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Save the currently displayed ROI image to the 'plates' folder with a unique filename
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        # Draw a filled rectangle at the bottom of the frame indicating the plate has been saved
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        # Add text indicating that the plate has been saved
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        # Display the updated frame
        cv2.imshow("Results", img)
        # Wait for 500 milliseconds before continuing
        cv2.waitKey(500)
        # Increment the counter for the next saved image
        count += 1
