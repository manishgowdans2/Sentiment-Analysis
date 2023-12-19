from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('EmotionDetectionModel.h5')

class_labels=['Angry','Happy','Neutral','Sad','Surprise']

cap=cv2.VideoCapture(0)

emotion_values = {emotion: [] for emotion in class_labels}
timestamps = []
max_history_length = 100  # Number of recent frames to keep in history

# Create an OpenCV window
cv2.namedWindow("Emotion Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion Analysis", 1000, 700)

plt.figure(figsize=(10, 6))

while True:
    ret,frame=cap.read()
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)

            preds=classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        predictions = classifier.predict(roi)


        emotion_values['Angry'].append(predictions[0][0])
        emotion_values['Happy'].append(predictions[0][1])
        emotion_values['Neutral'].append(predictions[0][2])
        emotion_values['Sad'].append(predictions[0][3])
        emotion_values['Surprise'].append(predictions[0][4])
       

        # Limit the history length to a fixed number of frames
        if len(timestamps) > max_history_length:
            for emotion in class_labels:
                emotion_values[emotion].pop(0)

        # Plot the line graph
        plt.clf()
        for emotion in class_labels:
            plt.plot(emotion_values[emotion], label=emotion)
        plt.title("Emotion Analysis")
        plt.xlabel("Frames")
        plt.ylabel("Emotion Probability")
        plt.legend()
        plt.grid()

        # Display approximate values for each emotion
        text_x = 10
        text_y = 40
        for emotion in class_labels:
            text = f"{emotion}: {emotion_values[emotion][-1]:.2f}"
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            text_y += 20

        plt.pause(0.01)
        plt.draw()
    
    cv2.imshow('Emotion Detector',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
cap.release()
cv2.destroyAllWindows()
