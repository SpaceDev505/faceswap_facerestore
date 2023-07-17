
from fastapi import FastAPI
import gradio as gr
import os
import uvicorn
from face_detection import select_face
from face_swap import face_swap

app = FastAPI()
@app.get('/test')
def test():
    return ('testapp, hello, how are you?')


import os
import cv2
import logging
import argparse

from face_detection import select_face
from face_swap import face_swap
from fgan.inference_gfpgan import main_inferene

height =0
width = 0
class VideoHandler(object):
    def __init__(self, video_path=0, img_path=None, args=None):
        self.src_points, self.src_shape, self.src_face = select_face(cv2.imread(img_path))
        if self.src_points is None:
            print('No face detected in the source image !!!')
            exit(-1)
        self.args = args
        self.video = cv2.VideoCapture(video_path)
        self.writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*'MJPG'), self.video.get(cv2.CAP_PROP_FPS),
                                      (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def start(self):
        while self.video.isOpened():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            _, dst_img = self.video.read()
            dst_points, dst_shape, dst_face = select_face(dst_img, choose=False)
            print(dst_points, dst_shape, dst_face)
            if dst_points is not None:
                dst_img = face_swap(self.src_face, dst_face, self.src_points, dst_points, dst_shape, dst_img, self.args, 68)
            self.writer.write(dst_img)
            if self.args.show:
                cv2.imshow("Video", dst_img)

        self.video.release()
        self.writer.release()
        cv2.destroyAllWindows()



def get_result(inputs_video, inputs_image):
    # src_points, src_shape, src_face = select_face(cv2.imread(inputs_image))
    src_points, src_shape, src_face = select_face(inputs_image)
    if src_points is None:
        print('No face detected in the source image !!!')
        exit(-1)
    print('123')
    save_path = './results/result.mp4'
    video = cv2.VideoCapture(inputs_video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), video.get(cv2.CAP_PROP_FPS),
    #                                   (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), video.get(cv2.CAP_PROP_FPS),
                                      (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while video.isOpened():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        success, dst_img = video.read()
        if success:
            # cv2.imshow('imge', dst_img)
            # cv2.waitK
            dst_points, dst_shape, dst_face = select_face(dst_img, choose=False)
            if dst_points is not None:
                dst_img = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, 68)
                cv2.imwrite('fgan/inputs/whole_imgs/image.jpg', dst_img)
                res = main_inferene()
                print("33333333", os.path.abspath(os.curdir))
                print(os.path.exists("test1.jpg"))
                img = cv2.imread('test1.jpg')
                img = cv2.resize(img, (width, height))
            writer.write(img) 
            # writer.write(dst_img)
        else:
            break
    video.release()
    writer.release()
    # cv2.destroyAllWindows()    
    return save_path

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Face Swap")
    with gr.Row():
        with gr.Column():
            inputs_video = gr.Video(label="Original Video", show_label=True)
            inputs_image = gr.Image(source='upload')
            with gr.Row():
                runBtn = gr.Button("FaceSwap")
        with gr.Column():
            # gallery = gr.Gallery(label="Generated images", show_label=False)
            result = gr.Video(label="Generated Video", show_label=True)
    runBtn.click(fn=get_result, inputs=[inputs_video, inputs_image], outputs=[result])

gr.mount_gradio_app(app, block, path='/')
if __name__ =="__main__":
    uvicorn.run(app, host="127.0.0.1" )

