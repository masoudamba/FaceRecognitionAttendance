import cv2
import numpy as np
import face_recognition

imageAmba = face_recognition.load_image_file('imageBasics/Amba.jpg')
imageAmba = cv2.cvtColor(imageAmba, cv2.COLOR_BGR2RGB)
imageTest = face_recognition.load_image_file('imageBasics/Amba Test.jpg')
imageTest = cv2.cvtColor(imageTest, cv2.COLOR_BGR2RGB)

# finding the faces in the image and the encoding

faceLoc = face_recognition.face_locations(imageAmba)[0]
encodeAmba = face_recognition.face_encodings(imageAmba)[0]
cv2.rectangle(imageAmba, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imageTest)[0]
encodeAmbaTest= face_recognition.face_encodings(imageTest)[0]
cv2.rectangle(imageTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

## Comparing these faces and finf=ding the distnce between them
results = face_recognition.compare_faces([encodeAmba], encodeAmbaTest)
##print the result to see if faes are similar
# if you change the picture to say like the bill gates picture it does the encoding fine but results are false since the contours do not match
# Sometimes images may be similar so the best thing to do is to find similarities using distances the lower the distance the best the match
faceDistance = face_recognition.face_distance([encodeAmba], encodeAmbaTest)
print(results, faceDistance)
cv2.putText(imageTest, f'{results} {round(faceDistance[0], 4)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Amba", imageAmba)
cv2.imshow("Amba Test", imageTest)
cv2.waitKey(0)

