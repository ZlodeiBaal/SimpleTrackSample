from sort import *
import random
import cv2

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def detect_cv2_camera(cfgfile, weightfile):
    #Заграузка сети
    net = cv2.dnn.readNetFromDarknet(cfgfile, weightfile)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #Инициализируем трекер
    mot_tracker = Sort()
    #Используем YOLO
    namesfile = 'coco.names'
    class_names = load_class_names(namesfile)
    #Камера
    cap = cv2.VideoCapture(0)
    #Генерим коробочкам разные цвета
    color_list = []
    for j in range(1000):
        color_list.append(((int)(random.randrange(255)),(int)(random.randrange(255)),(int)(random.randrange(255))))

    #go!
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            #Запустим сеть по картинке
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)

            h,w,_ = img.shape
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            #Разберём все выходы
            boxes=[]
            confidences=[]
            classIDs=[]
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                                    0.3)


            result_img = np.copy(img)
            dets = []
            count_detection=0
            for j in range(len(idxs)):
                name = class_names[classIDs[idxs[j][0]]]
                if name == 'person':
                    count_detection+=1
            if count_detection>0:
                detects = np.zeros((count_detection,5))
                count=0
                #Подготовим в формат который ест трекер
                for j in range(len(idxs)):
                    b = boxes[j]
                    name = class_names[classIDs[idxs[j][0]]]
                    if name == 'person':
                        x1 = int(b[0])
                        y1 = int(b[1])
                        x2 = int((b[0] + b[2]))
                        y2 = int((b[1] + b[3]))
                        box = np.array([x1,y1,x2,y2,confidences[idxs[j][0]]])
                        detects[count,:] = box[:]
                        count+=1
                #Подаём в трекер и сразу имеем результат!
                if len(detects)!=0:
                    trackers = mot_tracker.update(detects)
                    for d in trackers:
                        result_img = cv2.rectangle(result_img, ((int)(d[0]), (int)(d[1])), ((int)(d[2]), (int)(d[3])), color_list[(int)(d[4])], 2)
            cv2.imshow('demo', result_img)
            cv2.waitKey(1)
    cap.release()



if __name__ == '__main__':
    detect_cv2_camera('yolov3-tiny.cfg', 'yolov3-tiny.weights')