import cv2
import numpy as np
import face_recognition
import time

imgVivek = face_recognition.load_image_file('StudentsImages/Vivek.jpg')
imgVivek = cv2.cvtColor(imgVivek,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('StudentsImages/Vivek Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc =  face_recognition.face_locations(imgVivek)[0]
encodeVivek = face_recognition.face_encodings(imgVivek)[0]
cv2.rectangle(imgVivek,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest =  face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeVivek],encodeTest)
faceDis = round(face_recognition.face_distance([encodeVivek], encodeTest)[0],2)

print(results, faceDis)
cv2.putText(imgTest,f'{results} {faceDis}', (faceLocTest[3],faceLocTest[0]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


print(faceLoc)
print(faceLocTest)

cv2.imshow('Vivek', imgVivek)
cv2.imshow('Vivek Test', imgTest)


cv2.waitKey(1)
time.sleep(30)
cv2.destroyAllWindows()