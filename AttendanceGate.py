import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import Image
from pymongo import MongoClient
import io
import base64

client = MongoClient('mongodb+srv://DadhichVivek:24TP5Ic9sTYyBHyT@cluster0.am5qqrc.mongodb.net/test?retryWrites=true&w=majority')
images = []
classNames = []
mylist  = client['test']['postmessages'].find()
for cl in mylist:
    curImg = np.array(Image.open(io.BytesIO(base64.b64decode(cl['selectedFile'][23:]))))[:, :, ::-1]
    images.append(curImg)
    classNames.append(cl['title'])
print(classNames)

def findEncodings(images):
    encodeList = []

    for img in images:
        try:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except Exception:
            print(img)
    return np.array(encodeList)

def markAttendance(name):
    with open('StudentsAttendence/Attendance.csv','r+') as f:
        myDataList = f.readlines()

        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dtString1 = now.strftime('%d:%B:%Y')
            f.writelines(f'\n{name},{dtString},{dtString1}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv.VideoCapture(0)

while True:
    sucess, img = cap.read()
    imgS = cv.resize(img, (0,0), None, 0.25,0.25 )
    imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
            cv.rectangle(img, (x1,y2-35), (x2,y2),(0,255,0), cv.FILLED)
            cv.putText(img, name, (x1+6, y2-6), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markAttendance(name)
            name = name + ' ' + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
          #  print(f'./attend/{name}.jpg')
            file = f'./attend/{name}.jpg'
            print(file)
            print(cv.imwrite(file, img))


    cv.imshow('Webcam', img)
    cv.waitKey(1)



