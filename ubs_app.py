# [theme]
# base="light"
# backgroundColor="#efe9e9"
# secondaryBackgroundColor="#ccd4e4"
# font="serif"

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from yolov5_files.detect import run
from pathlib import Path
import json
import boto3
import cv2
import math
import moviepy.editor as moviepy
from converter import Converter


st.set_page_config(layout='wide')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


yolo_path = './'

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result
    
# Returns the latest folder in a runs\detect
def get_detection_folder():
    return max(get_subdirs(os.path.join('yolov5_files', 'runs', 'detect')), key=os.path.getmtime)


def analyzeVideo(videoFile):
    projectVersionArn = "arn:aws:rekognition:us-west-2:487699691653:project/Hotel_property_inspection_cv/version/Hotel_property_inspection_cv.2022-05-19T14.37.48/1652951268313"

    rekognition = boto3.client('rekognition', region_name='us-west-2')
    customLabels = []
    cap = cv2.VideoCapture(videoFile)
    print('No. of frames in the video: ', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frameRate = 1 # cap.get(cv2.CAP_PROP_FPS) # cap.get(5)  # frame rate
    print('Frame rate: ', frameRate)
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        print("Processing frame id: {}".format(frameId))
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            hasFrame, imageBytes = cv2.imencode(".jpg", frame)

            if (hasFrame):
                response = rekognition.detect_custom_labels(
                    Image={
                        'Bytes': imageBytes.tobytes(),
                    },
                    ProjectVersionArn=projectVersionArn
                )

            for elabel in response["CustomLabels"]:
                elabel["Timestamp"] = (frameId / frameRate) * 1000
                customLabels.append(elabel)

    print(customLabels)
    
    with open(os.path.join("rekog_output", videoFile.split('/')[-1].split('.')[0] + ".json"), "w") as f:
    # with open(videoFile.split('/')[-1].split('.')[0] + ".json", "w") as f:
        f.write(json.dumps(customLabels))

    cap.release()


# plotting function
def plotting(jsonFile, video):
    # plotting the labels onto the original video
    with open(jsonFile, 'r') as f:
        data = json.load(f)
    print(len(data))

    def get_data(dat, width, height):
        class_name = dat['Name']
        confidence = dat['Confidence']
        if confidence > 0.5:
            box = dat['Geometry']['BoundingBox']
            left = width * box['Left']
            top = height * box['Top']
            w = width * box['Width']
            h = height * box['Height']
            return [int(left), int(top), int(w), int(h)], float(confidence), class_name


    count = 0
    frame = 0
    writer = None
    vs = cv2.VideoCapture(video)
    OUTPUT_FILE = os.path.join("rekog_output", video.split('/')[-1].split('.')[0] + ".mp4")

    while 1:
        (grabbed, img) = vs.read()
        if not grabbed:
            print("not")
            break
        (height, width) = img.shape[:2]
        # print(frame)
        if frame in [int(i['Timestamp'] / 1000) for i in data]:
            boxes, conf, class_name = get_data(data[count], width, height)
            count = count + 1
            left, top, w, h = boxes
            # print(boxes)
            if len(boxes) > 0:
                # cv2.rectangle(img, (left+20, top-20), (left+w+40, top+h-20), (255, 0, 0), 2)
                cv2.rectangle(img, (left, top), (left + w, top + h), (255, 0, 0), 2)
                cv2.putText(img, class_name + " " + str(round(conf, 2)), (left, top-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        frame = frame + 1

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"avc1") #MJPG
            writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30, (img.shape[1], img.shape[0]), True)
        writer.write(img)

    writer.release()
    vs.release()


def video_convert(model):
    if model == 'v5':
        for vid in os.listdir(get_detection_folder()):
            path = str(Path(f'{get_detection_folder()}') / vid)
            new_path = path.replace('.mp4', '_new.mp4')
            
            vc = cv2.VideoCapture(path)
            if not vc.isOpened():
                print('Error: can not opencv camera')
                exit(0)

            ret, frame = vc.read()
            w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vc.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            vw = cv2.VideoWriter(new_path, fourcc, fps, (w, h), True)
            while ret:
                vw.write(frame)
                ret, frame = vc.read()
            vw.release()
            
    elif model == 'rekog':
        for vid in os.listdir('rekog_output'):
            if vid.split('.')[-1] == 'mp4':
                path = str(Path('rekog_output') / vid)
                new_path = path.replace('.mp4', '_new.mp4')
                
                vc = cv2.VideoCapture(path)
                if not vc.isOpened():
                    print('Error: can not opencv camera')
                    exit(0)

                ret, frame = vc.read()
                w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = vc.get(cv2.CAP_PROP_FPS)

                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                vw = cv2.VideoWriter(new_path, fourcc, fps, (w, h), True)
                while ret:
                    vw.write(frame)
                    ret, frame = vc.read()
                vw.release()
            
            
            
# main function of the app
def main():
    print(str(Path('rekog_output') / 'vid'))
    for vid in os.listdir(get_detection_folder()):
        path = str(Path(f'{get_detection_folder()}') / vid)
        new_path = path.replace('.mp4', '_new.mp4')
        print(path, new_path)

    st.title("Property Inspection(Object Detection)")
    
    source = ("Yolov5", "AWS-Rekognition")
    source_index = st.sidebar.selectbox("Select the model", range(len(source)), format_func=lambda x: source[x]) 
    
    model = False
    is_valid = False
 
    
    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader("Upload video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='loading...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("yolov5_files", "data1", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # picture = picture.save(f'yolov5_files/data1/images/{uploaded_file.name}')
                weightsPath = os.path.sep.join([yolo_path, "yolov5_weights/best.pt"])
                model = 'v5'
        else:
            is_valid = False 
            
    else:
        st.write("You can view the property inspection using AWS Rekognition here:")
        uploaded_file = st.sidebar.file_uploader("Upload video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='loading...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("yolov5_files", "data1", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # picture = picture.save(f'yolov5_files/data1/images/{uploaded_file.name}')
                # weightsPath = os.path.sep.join([yolo_path, "yolov5_multiple/best.pt"])
                model = 'rekog'
        else:
            is_valid = False        
            
    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)         

    if is_valid:
        print('valid')
        if st.button('detect'):
        
            if model == 'rekog':
                with st.spinner(text='Analysing the video'):
                    analyzeVideo(f'yolov5_files/data1/videos/{uploaded_file.name}')
                    plotting(os.path.join("rekog_output", uploaded_file.name.split('.')[0] + ".json"), f'yolov5_files/data1/videos/{uploaded_file.name}')
                    with st.spinner(text='Preparing Video'):
                        for vid in os.listdir('rekog_output'):
                            if vid.split('.')[-1] == 'mp4':
                                path = str(Path('rekog_output') / vid)
                                new_path = path.replace('.mp4', '_new.mp4')
                                video_convert(model)
                                st.video(new_path)
                    
            elif model == 'v5':
                with st.spinner(text='Preparing Video'):
                    img = run(weights=weightsPath, conf_thres=score_threshold, source=f'yolov5_files/data1/videos/{uploaded_file.name}')                    
                    with st.spinner(text='Preparing Video'):
                        for vid in os.listdir(get_detection_folder()):
                            path = str(Path(f'{get_detection_folder()}') / vid)
                            new_path = path.replace('.mp4', '_new.mp4')
                            video_convert(model)
                            st.video(new_path)
   
if __name__ == '__main__':
    main()

