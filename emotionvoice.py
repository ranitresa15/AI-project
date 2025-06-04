import cv2
from deepface import DeepFace
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
last_emotion = None
cap=cv2.VideoCapture(0)
while True:
    key,img=cap.read()
    results=DeepFace.analyze(img,actions=['emotion'],enforce_detection=False)
    emotion=results[0]['dominant_emotion']
    cv2.putText(img, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    if emotion != last_emotion:
        engine.say(f"The detected emotion is {emotion}")
        engine.runAndWait()
        last_emotion = emotion
    cv2.imshow("Emotion Recognition",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
