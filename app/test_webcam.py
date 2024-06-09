import cv2

cap = cv2.VideoCapture(0)  # Essayez d'autres indices comme 1, 2, etc. si 0 ne fonctionne pas
if not cap.isOpened():
    print("Error: Could not open video device")
else:
    print("Success: Camera opened")
    cap.release()
