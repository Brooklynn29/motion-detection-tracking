import cv2
import numpy as np

cap = cv2.VideoCapture("vid.avi")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
saved_frame = cv2.VideoWriter("motion_detection_tracking.avi", fourcc, 20.0, (frame_width, frame_height))

ret1, frame1 = cap.read()
ret2, frame2 = cap.read()

kernel = np.ones((3, 3), np.uint8)

while cap.isOpened():

    diff = cv2.absdiff(frame1, frame2)  # Difference between two frames (frame1 and frame2)
    gray_frame = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # Convert the difference frame to gray scale
    g_blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)  # Apply blurring
    _, th = cv2.threshold(g_blur, 20, 255, cv2.THRESH_BINARY)  # Apply THRESHOLD
    dilated_frame = cv2.dilate(th, kernel=kernel, iterations=3)  # Dilation (morphology)
    contours, _ = cv2.findContours(dilated_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding contours

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 1100:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, f"Status: Movement", (10, 20), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 2)

    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 3)

    saved_frame.write(frame1)
    cv2.imshow("result", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) & 0xFF == 27:
        break

cap.release()
saved_frame.release()
cv2.destroyAllWindows()

