import depthai as dai
import cv2
import numpy as np

import flask
from flask import Flask, render_template, Response, request,jsonify
from flask_cors import CORS
from io import BytesIO
import threading
from pathlib import Path
import time
import yaml
from math import *
from scipy.spatial.transform import Rotation
import requests
import pandas as pd

''' ====================================調用鏡頭的前置定義程序========================================'''

# 定義管道
# pipeline = dai.Pipeline()
# #創建攝像頭節點
# mono = pipeline.createMonoCamera()
# #選擇相機
# mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
# #創建XLinkOut節點 並獲取幀
#
# xout = pipeline.createXLinkOut()
# xout.setStreamName("left")
# mono.out.link(xout.input)

''' ============================================================================'''

app = flask.Flask(__name__)

# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
#
CORS(app)
# app.config["DEBUG"] = True
# app.config["JSON_AS_ASCII"] = False

image_queue = []        #影像
image_queue_buffer = [] #影像緩存
labels_api = []         #標偵測資訊
threshold_value = 50    #閾值
terminate_thread = False #啟動/關閉執行序
Auto_Mode_switch = True #自動模式

Take_photo = False      #拍照       (校正用-資料夾416x416_calibration)
robot_coord = []        #手臂記錄座標(校正用-資料夾416x416_calibration)
#相機偵測線程


def process_images():#OAK-D
    global image_queue
    global image_queue_buffer
    global terminate_thread
    global labels_api
    global threshold_value
    global Auto_Mode_switch

    # nnBlobPath = str(
    #     (Path(__file__).parent / Path('yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    mpath = "yolov5n_openvino_2021.4_6shave.blob"
    nnBlobPath = str((Path(__file__).parent / Path(mpath)).resolve().absolute())
    # nnBlobPath = str((Path(__file__).parent / Path('../models/person-detection-retail-0013_openvino_2021.4_7shave.blob')).resolve().absolute())
    # if 1 < len(sys.argv):
    #     arg = sys.argv[1]
    #     if arg == "yolo3":
    #         nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    #     elif arg == "yolo4":
    #         nnBlobPath = str((Path(__file__).parent / Path('../models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    #     else:
    #         nnBlobPath = arg
    # else:
    #     print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")
    if not Path(nnBlobPath).exists():
        import sys
        raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

    # Tiny yolo v3/4 label texts
    # labelMap = [
    #     "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    #     "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    #     "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    #     "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    #     "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    #     "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    #     "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    #     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    #     "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    #     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    #     "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    #     "teddy bear", "hair drier", "toothbrush"
    # ]
    labelMap = [
        "test",
    ]
    # RT_chess_to_base = end_to_base @ cam2end
    # print("相機座標至末端座標值 : ",
    #       RT_chess_to_base @ [round(tvecs[0][0], 2) * 10, round(tvecs[1][0] * 10, 2), round(tvecs[2][0] * 10, 2), 1])
    # print("棋盤格座標轉換為機械手臂座標 : ",
    #       RT_chess_to_base @ [round(tvecs[0][0], 2) * 10, round(tvecs[1][0] * 10, 2), round(tvecs[2][0] * 10, 2), 1])

    syncNN = True

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    nnNetworkOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutDepth.setStreamName("depth")
    nnNetworkOut.setStreamName("nnNetwork")

    # Properties
    camRgb.setPreviewSize(416, 416)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

    spatialDetectionNetwork.setBlobPath(nnBlobPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(1)
    spatialDetectionNetwork.setCoordinateSize(4)
    # spatialDetectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
    spatialDetectionNetwork.setAnchors([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326])

    # spatialDetectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
    spatialDetectionNetwork.setAnchorMasks({ "side52": [0, 1, 2],
    "side26": [3, 4, 5],
    "side13": [6, 7, 8]})
    spatialDetectionNetwork.setIouThreshold(0.5)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    camRgb.preview.link(spatialDetectionNetwork.input)

    if syncNN:
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
    spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        printOutputLayersOnce = True

        while True:
            if terminate_thread:
                if Auto_Mode_switch:
                    inPreview = previewQueue.get()
                    inDet = detectionNNQueue.get()
                    depth = depthQueue.get()
                    inNN = networkQueue.get()

                    if printOutputLayersOnce:
                        toPrint = 'Output layer names:'
                        for ten in inNN.getAllLayerNames():
                            toPrint = f'{toPrint} {ten},'
                        print(toPrint)
                        printOutputLayersOnce = False;

                    frame = inPreview.getCvFrame()
                    frames = frame.copy()

                    depthFrame = depth.getFrame()  # depthFrame values are in millimeters

                    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depthFrameColor = cv2.equalizeHist(depthFrameColor)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

                    counter += 1
                    current_time = time.monotonic()
                    if (current_time - startTime) > 1:
                        fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time

                    detections = inDet.detections

                    # If the frame is available, draw bounding boxes on it and show the frame
                    height = frame.shape[0]
                    width = frame.shape[1]
                    a = 0
                    labels_api = []
                    for detection in detections:
                        confidence = round(detection.confidence * 100, 2)  # 信心度
                        if confidence >= threshold_value:
                            # print(detection.confidence)
                            a+=1
                            roiData = detection.boundingBoxMapping
                            roi = roiData.roi
                            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                            topLeft = roi.topLeft()
                            bottomRight = roi.bottomRight()
                            xmin = int(topLeft.x)
                            ymin = int(topLeft.y)
                            xmax = int(bottomRight.x)
                            ymax = int(bottomRight.y)
                            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                            # Denormalize bounding box
                            x1 = int(detection.xmin * width)
                            x2 = int(detection.xmax * width)
                            y1 = int(detection.ymin * height)
                            y2 = int(detection.ymax * height)
                            try:
                                label = labelMap[detection.label]
                            except:
                                label = detection.label

                            """
                            Bounding Box 區域
                            """
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                            # 設定透明度
                            alpha = 0.4
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (250, 0, 0), -1)  # 區內上色
                            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                            """
                             字體 區域
                            """
                            x_offset = 0
                            y_offset = 50#調整整體高度
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 0.4
                            thickness = 1
                            labelSize = cv2.getTextSize(label, fontFace, fontScale, thickness)
                            _x2 = x1 + labelSize[0][0]  # topright x of text
                            _y2 = y1 - labelSize[0][1]  # topright y of text

                            cv2.rectangle(frame, (x1, y1 + y_offset), (_x2 + 50, _y2 + y_offset + 30), (0, 255, 0),
                                          cv2.FILLED)  # 畫text內框
                            cv2.rectangle(frame, (x1, y1 + y_offset + 15), (_x2 + 50, _y2 + y_offset + 90),
                                          (255, 255, 255), -1)  # 畫全部外框
                            cv2.putText(frame, str(a), (x1 + 4, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.3, 125)

                            cv2.putText(frame, str(label), (x1+18 , y1+ y_offset+10 ), cv2.FONT_HERSHEY_TRIPLEX, 0.3, 125)

                            cv2.putText(frame, f"{confidence}", (x1 , y1 + y_offset + 25),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, 255)
                            cv2.putText(frame, f"X:{int(detection.spatialCoordinates.x)} mm", (x1 , y1 + y_offset + 40),
                                        cv2.FONT_HERSHEY_TRIPLEX, .3, (0, 200, 0))
                            cv2.putText(frame, f"Y:{int(detection.spatialCoordinates.y)} mm", (x1 , y1 + y_offset + 55),
                                        cv2.FONT_HERSHEY_TRIPLEX, .3, (0, 200, 0))
                            cv2.putText(frame, f"Z:{int(detection.spatialCoordinates.z)} mm", (x1 , y1 + y_offset + 70),
                                        cv2.FONT_HERSHEY_TRIPLEX, .3, (0, 200, 0))

                            cv2.putText(frame, f"Find: {a} ", (300, 410),
                                    cv2.FONT_HERSHEY_TRIPLEX, .5, (100, 100, 100))


                            labels_api.append({0: [f"{str(a)} - {str(label)}" ,f" {confidence}% , "
                                                   f"X: {int(detection.spatialCoordinates.x)} mm,"
                                                   f"Y: {int(detection.spatialCoordinates.y)} mm,"
                                                   f"Z: {int(detection.spatialCoordinates.z)} mm"]})
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
                            # print(f"信心度{a}",confidence)
                    depthFrameColor = cv2.resize(depthFrameColor, (416, 416))
                    # print(depthFrameColor.shape[1])
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
                    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                                color)
                    ct = time.time()
                    cv2.putText(frame, "Time %s.%03d" % (time.strftime('%H:%M:%S'),(ct-int(ct))*1000), (120, 410),
                                cv2.FONT_HERSHEY_TRIPLEX, .5, (100, 100, 250))
                    imOut = np.hstack((depthFrameColor, frame))
                    # imOut = np.hstack((imOut, frame))
                    cv2.waitKey(1)


                else:
                    break
            else:
                break
            image_queue.append(imOut)
            image_queue_buffer = imOut

            # image_queue = imOut



def process_calibration():
    global image_queue
    global terminate_thread
    global image_queue_buffer
    global Take_photo

    pipeline = dai.Pipeline()
    # 定義取用RGB相機
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(416, 416)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam_rgb.setInterleaved(False)
    # XLinkOut 是設備的“輸出”。您要傳輸到主機的任何數據都需要通過 XLink 發送
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    # 將相機輸出串流畫面端命名為rgb
    xout_rgb.setStreamName("rgb")
    # 將上面定義的RGB相機預覽輸入至XLinkOut，以便將幀發送至主機
    cam_rgb.preview.link(xout_rgb.input)
    check_times = 0
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")

        while True:
            if terminate_thread:
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()
                if Take_photo:
                    check_times +=1
                    cv2.imwrite(f'416x416_calibration/img/calibration{check_times}.jpg', frame)
                    Take_photo = False
            else:
                break
            image_queue.append(frame)
            image_queue_buffer = frame


def generate():
    global image_queue
    global terminate_thread
    global image_queue_buffer
    while True:
        if not image_queue:
            continue
        # Get the latest processed image from the queue
        if terminate_thread:
            if image_queue != []:
                frame = image_queue.pop()
                # Encode the image to JPEG format
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_queue = []
        else:
            if image_queue != []:
                try:
                    frame = image_queue[0]
                    # Encode the image to JPEG format
                    _, img_encoded = cv2.imencode('.jpg', frame)
                except:
                    print("wrong")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

#用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R

#用于根据位姿计算变换矩阵
def pose_robot(x, y, z, Tx, Ty, Tz):
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    return RT1

#輸出偵測畫面
@app.route('/screen', methods=['GET'])
def video_feed():
    global ban
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#輸出偵測資料
@app.route('/screen3', methods=['GET'])
def video_feed3():
    global ban
    global labels_api
    return jsonify(labels_api)

#計算反饋手臂座標
@app.route('/robotcoord', methods=['POST'])
def robotcoord():
    global ban
    global labels_api
    data_res = request.get_json()
    print(int(data_res['camx']))
    print(int(data_res['camy']))
    print(int(data_res['camz']))
    camx = int(data_res['camx'])
    camy = int(data_res['camy'])
    camz = int(data_res['camz'])
    x = float(data_res['coordx'])
    y = float(data_res['coordy'])
    z = float(data_res['coordz'])
    rx = float(data_res['coordw'])
    ry = float(data_res['coordp'])
    rz = float(data_res['coordr'])
    print(x,y,z,rx,ry,rz)

    # 載入相機相對於末端的變換矩陣 - 手眼標定結果
    with open('camera_to_end_20230911_142411.yaml', 'r') as f:
        cam2end_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    # 相機到末端
    cam2end = np.array(cam2end_data["camera_to_end"], dtype=np.float64)
    print(cam2end)
    print("相機到末端",cam2end @ [0,0,0,1])
    print("相機到末端", np.dot(cam2end, [0, 0, 0, 1]))
    # 末端到基座
    end_to_base = pose_robot(rx, ry, rz, x, y, z)
    RT_chess_to_base = end_to_base @ cam2end

    # print("相機座標至末端座標值 : ",
    #       RT_chess_to_base @ [round(tvecs[0][0], 2) * 10, round(tvecs[1][0] * 10, 2), round(tvecs[2][0] * 10, 2), 1])
    print("棋盤格座標轉換為機械手臂座標 : ",
          RT_chess_to_base @ [camx , camy, camz, 1])
    posecoord = RT_chess_to_base @ [camx , camy, camz, 1]
    posex = round(posecoord[0],3)
    posey = round(posecoord[1],3)
    posez = round(posecoord[2],3)
    print(posex,posey,posez)
    url = 'http://192.168.2.105:8081/RoboControl.aspx/Web_SetCoords'

    # 準備要發送的數據，這裡假設您要傳遞一個JSON對象
    data = {'cordX': posex, 'cordY': posey, 'cordZ': posez, 'cordW': "", 'cordP': "", 'cordR': ""}

    # 使用requests庫發送POST請求
    response = requests.post(url, json=data, verify=False)

    # 檢查請求是否成功
    if response.status_code == 200:
        print('POST請求成功')
        print('伺服器回應：', response.text)
    else:
        print('POST請求失敗')

    return Response("ff")
#調整閾值
@app.route('/threshold', methods=['POST'])
def threshold():
    global threshold_value
    data_res = request.get_json()
    if data_res != "":
        threshold_value = int(data_res['threshold'])
    print(threshold_value)
    return jsonify({f"setoff":1})
#關閉相機並保存最後一幀
@app.route('/close', methods=['GET'])
def close_screen():
    global terminate_thread
    global processing_thread
    global image_queue_buffer
    global image_queue
    global robot_coord

    robot_coord = []#清空robot紀錄
    print("關閉後",image_queue_buffer)
    image_queue.append(image_queue_buffer)
    print("關閉後 原始",image_queue)
    ct = time.time()
    print("測試")
    terminate_thread = False
    return jsonify({f"turn off done": "%s.%03d" % (time.strftime('%H:%M:%S'),(ct-int(ct))*1000)})

#打開偵測
@app.route('/open', methods=['GET'])
def open_screen():
    global terminate_thread
    global image_queue_buffer
    image_queue_buffer = []
    print("開啟後",image_queue_buffer)

    terminate_thread = True
    processing_thread = threading.Thread(target=process_images)
    processing_thread.daemon = True
    processing_thread.start()
    return Response("done")

#打開校正相機
@app.route('/calibration', methods=['GET'])
def calibration():
    global terminate_thread
    global image_queue_buffer
    image_queue_buffer = []
    print("開啟後",image_queue_buffer)

    if not terminate_thread:
        terminate_thread = True
        processing_thread = threading.Thread(target=process_calibration)
        processing_thread.daemon = True
        processing_thread.start()

    return Response("done")

@app.route('/openAutoMode', methods=['POST'])
def openAutoMode():
    global terminate_thread
    global image_queue_buffer
    global labels_api
    global Auto_Mode_switch

    image_queue_buffer = []
    print("自動模式開啟後",terminate_thread)
    print("自動模式開啟後",Auto_Mode_switch)

    if not terminate_thread:
        terminate_thread = True
        processing_thread = threading.Thread(target=process_images)
        processing_thread.daemon = True
        processing_thread.start()
    if labels_api != []:
        Auto_Mode_switch = False
        data_res = request.get_json()

        x = float(data_res['coordx'])
        y = float(data_res['coordy'])
        z = float(data_res['coordz'])
        rx = float(data_res['coordw'])
        ry = float(data_res['coordp'])
        rz = float(data_res['coordr'])

        for obj in range(len(labels_api)):
            data = labels_api.pop(0)
            print("ALL==========",data[0][1])

            camx = int(data[0][1].split(',')[1][3:-3])
            camy = int(data[0][1].split(',')[2][3:-3])
            camz = int(data[0][1].split(',')[3][3:-3])
            if camz > 1000:
                break
            print(camx,camy,camz)

            with open('camera_to_end_20230911_142411.yaml', 'r') as f:
                cam2end_data = yaml.load(f.read(), Loader=yaml.FullLoader)
            # 相機到末端
            cam2end = np.array(cam2end_data["camera_to_end"], dtype=np.float64)
            print("相機到末端", cam2end @ [0, 0, 0, 1])
            print("相機到末端", np.dot(cam2end, [0, 0, 0, 1]))
            # 末端到基座
            end_to_base = pose_robot(rx, ry, rz, x, y, z)
            RT_chess_to_base = end_to_base @ cam2end

            # print("相機座標至末端座標值 : ",
            #       RT_chess_to_base @ [round(tvecs[0][0], 2) * 10, round(tvecs[1][0] * 10, 2), round(tvecs[2][0] * 10, 2), 1])
            print("棋盤格座標轉換為機械手臂座標 : ",
                  RT_chess_to_base @ [camx, camy, camz, 1])
            posecoord = RT_chess_to_base @ [camx, camy, camz, 1]
            posex = round(posecoord[0], 3)
            posey = round(posecoord[1], 3)
            posez = round(posecoord[2], 3)
            print(posex, posey, posez)
            url = 'http://192.168.2.105:8081/RoboControl.aspx/Web_SetCoords'

            # 準備要發送的數據，這裡假設您要傳遞一個JSON對象
            data = {'cordX': posex, 'cordY': posey, 'cordZ': posez, 'cordW': "", 'cordP': "", 'cordR': ""}

            # 使用requests庫發送POST請求
            response = requests.post(url, json=data, verify=False)

            # 檢查請求是否成功
            if response.status_code == 200:
                print('POST請求成功')
                print('伺服器回應：', response.text)
            else:
                print('POST請求失敗')
        print(labels_api)
        Auto_Mode_switch = True

        # print(len(labels_api))
    return jsonify({f"turn off done":labels_api})

    # return Response(labels_api)

#拍照+紀錄座標
@app.route('/take', methods=['POST'])
def take():
    global Take_photo
    global robot_coord
    Take_photo = True
    data_res = request.get_json()
    robot_coord.append([data_res['cordX'],
                        data_res['cordY'],
                        data_res['cordZ'],
                        data_res['cordW'],
                        data_res['cordP'],
                        data_res['cordR'],
                        ])
    print(robot_coord)
    df = pd.DataFrame(robot_coord, columns=['dx', 'dy', 'dz', 'ax', 'ay', 'az'])
    df.to_csv('416x416_calibration/pos/pos.csv', index=False)
    # Get the latest processed image from the queue
    return Response("done")

#紀錄座標
@app.route('/record_coord', methods=['GET'])
def record_coord():
    global Take_photo
    global robot_coord
    return jsonify({f"coord": robot_coord})

@app.route('/', methods=['GET'])
def index():
    return Response("none")

if __name__ == '__main__':
    # process_images()
    # processing_thread = threading.Thread(target=process_images)
    # processing_thread.daemon = True
    # processing_thread.start()
    app.run(host='192.168.2.105', ssl_context=('server.crt', 'server.key'), threaded=True)

    # app.run(host='0.0.0.0',threaded=True )


