
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time

###############################
classID = 0 # 0 is fake 1 is real
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 35 # larger is more focus

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
###############################

cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4, camHeight)
detector = FaceDetector()

"""Run the loop to continually get frames from the webcam"""
while True:
    # Read the current frame from the webcam
    # success: Boolean, whether the frame was successfully grabbed
    # img: the captured frame
    success, img = cap.read()
    imgOut = img.copy()
    # Detect faces in the image
    # img: Updated image
    # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.findFaces(img, draw=False)


    listBlur = []#True False Values indicating if the faces are blur or not
    listInfo = [] # The normalize values and the class name for the label txt file

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]
            #print(x, y, w, h)

            # --- check the score -----
            if score > confidence:
                ###---Adding an Offset to the Face detector-------

                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetW * 3.5)

                # ---- To avoid values below 0 ------

                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0


                # ---- Find Bluriness -----

                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue>blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)


                # --------Normalize values --------
                ih, iw, _ = img.shape
                xc, yc = x + w/2, y+h/2

                xcn, ycn = round(xc/iw,floatingPoint), round(yc/ih,floatingPoint)
                wn, hn = round(w/iw,floatingPoint), round(h/ih,floatingPoint)
                print(xcn, ycn,wn, hn)

                # ---- To avoid values above 1 ------

                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")


                # -- Drawing -----
                cv2.rectangle(imgOut, (x,y, w,h), (255,0,0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score*100)}% Blur:{blurValue}', (x, y - 20),
                                   scale=2, thickness= 3)

                if debug:
                    cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur:{blurValue}', (x, y - 20),
                                       scale=2, thickness=3)

        # ---------To Save --------------------
        if save:
            if all(listBlur) and listBlur!=[]:
                # ---- Save image ------
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                # print(timeNow)
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                # -- Save Label Text file-----------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()
                



    # Display the image in a window named 'Image'
    cv2.imshow("Image", imgOut)
    # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)

