import cv2
from deepface import DeepFace
import numpy as np

known_face = "test.jpg"
known_embedding = DeepFace.represent(img_path=known_face,model_name="VGG-Face")[0]["embedding"]
known_embedding = np.array(known_embedding)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(2)
threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Only compute embedding if a face is detected
            unknown_embedding = DeepFace.represent(img_path=face_roi,model_name="VGG-Face", enforce_detection=False)
            if len(unknown_embedding) > 0:
                unknown_embedding = np.array(unknown_embedding[0]["embedding"])
                cos_sim = np.dot(known_embedding, unknown_embedding) / (np.linalg.norm(known_embedding) * np.linalg.norm(unknown_embedding))
                print("Cosine Similarity:", cos_sim)

                if cos_sim > threshold:
                    cv2.putText(frame, "Same Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Different Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error: {e}")

    cv2.imshow("Real-Time Face Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 