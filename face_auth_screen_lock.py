import cv2
import torch
import numpy as np
import os
import time
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load admin face embedding
admin_img = cv2.imread("D:\\Face-recognition\\ashish.jpg")
admin_face = cv2.cvtColor(admin_img, cv2.COLOR_BGR2RGB)
admin_face_tensor = mtcnn(admin_face)
admin_embedding = resnet(admin_face_tensor.unsqueeze(0).to(device))

def lock_screen():
    os.system('rundll32.exe user32.dll,LockWorkStation')

# Start webcam
cap = cv2.VideoCapture(0)
not_detected_start = None
grace_period = 1.5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(rgb_frame)

    now = time.time()

    if face_tensor is None:
        if not_detected_start is None:
            not_detected_start = now
        elif now - not_detected_start >= grace_period:
            lock_screen()
    else:
        face_embedding = resnet(face_tensor.unsqueeze(0).to(device))
        distance = (admin_embedding - face_embedding).norm().item()

        if distance > 0.8:
            if not_detected_start is None:
                not_detected_start = now
            elif now - not_detected_start >= grace_period:
                lock_screen()
        else:
            not_detected_start = None  # Admin recognized, reset timer

    cv2.imshow('Face Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()