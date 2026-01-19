import cv2
import face_recognition
import pickle
import os

# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])
print(studentIds)
def findEncodings(imagesList):
    #Loop through all images and generate the encodings for the same
    encodeList = []
    for img in imagesList:
        # open cv uses bgr and face-recognition uses rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
print(encodeListKnownWithIds)
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")