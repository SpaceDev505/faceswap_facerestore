import gradio as gr
import os
from face_detection import select_face, select_all_faces
from face_swap import face_swap
import cv2

def process(input_src_image, input_tar_image):

    # Select src face
    src_points, src_shape, src_face = select_face(input_src_image)
    # Select dst face
    dst_faceBoxes = select_all_faces(input_tar_image)

    if dst_faceBoxes is None:
        print('Detect 0 Face !!!')
        exit(-1)

    output = input_tar_image
    for k, dst_face in dst_faceBoxes.items():
        output = face_swap(src_face, dst_face["face"], src_points,
                           dst_face["points"], dst_face["shape"],
                           output)

    result = output

    return result

def _main ():
    block = gr.Blocks().queue()
    with  block:
        with gr.Row():
            gr.Markdown("FaceSwap")
        with gr.Row():
            with gr.Column():
                input_src_image = gr.Image(source='upload')
                input_tar_image = gr.Image(source='upload')
                with gr.Row():
                    swapBtn = gr.Button('Swap')
            with gr.Column():
                result = gr.Image()
        swapBtn.click(fn=process, inputs=[input_src_image, input_tar_image], outputs=[result])
    block.launch(share=True)

if __name__ == "__main__":
    _main()

