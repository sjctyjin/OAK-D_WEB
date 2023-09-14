import depthai as dai
import cv2
import numpy as np

import flask
from flask import Flask, render_template, Response,request
from flask_cors import CORS
from io import BytesIO
import threading

''' ====================================調用鏡頭的前置定義程序========================================'''

#定義管道
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

image_queue = []
terminate_thread = True



def process_images():
    global image_queue
    global terminate_thread

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
    #
    # syncNN = True
    # 定義管道
    pipeline = dai.Pipeline()
    # 定義取用RGB相機
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    # cam_rgb2 = pipeline.create(dai.node.ColorCamera)
    # cam_rgb3 = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(416, 416)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam_rgb.setInterleaved(False)
    #定義深度
    # spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    # monoLeft = pipeline.create(dai.node.MonoCamera)
    # monoRight = pipeline.create(dai.node.MonoCamera)
    # stereo = pipeline.create(dai.node.StereoDepth)
    # nnNetworkOut = pipeline.create(dai.node.XLinkOut)
    """
    # 不同設定格式
    # cam_rgb2.setPreviewSize(1920, 1080)
    # cam_rgb2.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    # cam_rgb2.setInterleaved(False)
    #
    # cam_rgb3.setPreviewSize(1920, 1080)
    # cam_rgb3.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # cam_rgb3.setInterleaved(False)
    
    """

    # XLinkOut 是設備的“輸出”。您要傳輸到主機的任何數據都需要通過 XLink 發送
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    # xoutNN = pipeline.create(dai.node.XLinkOut)
    # xoutDepth = pipeline.create(dai.node.XLinkOut)

    """
    #不同輸出格式
    # xout_rgb2 = pipeline.create(dai.node.XLinkOut)
    # xout_rgb3 = pipeline.create(dai.node.XLinkOut)
    """

    # 將相機輸出串流畫面端命名為rgb
    xout_rgb.setStreamName("rgb")
    # xoutNN.setStreamName("detections")
    # xoutDepth.setStreamName("depth")
    # nnNetworkOut.setStreamName("nnNetwork")
    """
    #命名其他相機串流類別
    # xout_rgb2.setStreamName("rgb2")
    # xout_rgb3.setStreamName("rgb3")
    """

    # 將上面定義的RGB相機預覽輸入至XLinkOut，以便將幀發送至主機
    cam_rgb.preview.link(xout_rgb.input)
    # cam_rgb2.preview.link(xout_rgb2.input)
    # cam_rgb3.preview.link(xout_rgb3.input)
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        # q_rgb2 = device.getOutputQueue("rgb2")
        # q_rgb3 = device.getOutputQueue("rgb3")
        while True:

            while terminate_thread:
                in_rgb = q_rgb.get()
                # in_rgb2 = q_rgb2.get()
                # in_rgb3 = q_rgb3.get()
                frame = in_rgb.getCvFrame()
                image_queue.append(frame)
            print(q_rgb)


def process_imagess():
    global image_queue
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        # Perform image processing on 'frame'
        # ...
        # Put the processed image in the queue
        image_queue.append(frame)

def generate():
    global image_queue
    while True:
        if not image_queue:
            continue
        # Get the latest processed image from the queue
        frame = image_queue.pop()
        image_queue = []
        # Encode the image to JPEG format
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

@app.route('/screen',methods=['GET'])
def video_feed():
    global ban
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/close',methods=['GET'])
def close_screen():
    global terminate_thread
    print("測試")
    terminate_thread = False
    return Response("turn off done")

@app.route('/open',methods=['GET'])
def open_screen():
    global terminate_thread
    terminate_thread = True
    return Response("done")
@app.route('/',methods=['GET'])
def index():
    return Response("none")


if __name__ == '__main__':
    # process_images()
    processing_thread = threading.Thread(target=process_images)
    processing_thread.daemon = True
    processing_thread.start()
    app.run(host='0.0.0.0',ssl_context=('server.crt', 'server.key'),threaded=True)

    # app.run(host='0.0.0.0',threaded=True )


