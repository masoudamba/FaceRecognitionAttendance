import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imageAttendance/'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    currentImage = cv2.imread(f'{path}/{cl}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        namesList = []
        for line in myDataList:
            entry = line.split(',')
            namesList.append(entry[0])
        if name not in namesList:
            now = datetime.now()
            dateStrFmt = now.strftime('%I:%M:%S %p')
            f.writelines(f'\n{name},{dateStrFmt}')


encodeGroupMembers = findEncodings(images)
# print(len(encode_group_members))  ##test if it works
print("Encoding complete")

# Initialize web cam to capture faces to compare to our data
cap = cv2.VideoCapture(0)

# while lop to capture each and every frame
while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing the image so as to run faster
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    facesCurrFrame = face_recognition.face_locations(imgSmall)  # Find all faces in the frame
    encodingCurrFrame = face_recognition.face_encodings(imgSmall, facesCurrFrame)

    for encode_frames, faces in zip(encodingCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeGroupMembers, encode_frames)
        faceDistance = face_recognition.face_distance(encodeGroupMembers, encode_frames)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)
        if matches[matchIndex]:
            name = str(classNames[matchIndex]).upper()
            print(name)
            y1, x2, y2, x1 = faces
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  ##multiply by 4 since we scaled it down
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1 + 6, y2 - 6), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            y1, x2, y2, x1 = faces
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  ##multiply by 4 since we scaled it down
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1 + 6, y2 - 6), (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, "Visitor", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
