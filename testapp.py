import gradio as gr
import os
from fastapi import FastAPI
from generate_audio import generate_audio
from inference_audio_image import inference2
from inference_audio_video import inference1


app = FastAPI()

def convert(segment_length, video, audio):
    return audio

def get_result(input_video, input_image):
    if input_video == None:
        result = inference2(input_image)
        return result
    elif input_image == None:
        result = inference1(input_video)
        return result
    
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Lip Sync")
    with gr.Row():
        with gr.Column():
            inputs_video = gr.Video(label="Original Video", show_label=True)
            inputs_image = gr.Image(source='upload')
            input_text = gr.Textbox(label="Please input text you want")
            audio = gr.outputs.Audio(type="numpy", label=None)
            with gr.Row():
                audioBtn = gr.Button(label="Generate Audio")
                runBtn = gr.Button(label="Run")
        with gr.Column():
            # gallery = gr.Gallery(label="Generated images", show_label=Fal>
            result = gr.Video(label = "Generated Video", show_label=True)
    audioBtn.click(fn=generate_audio, inputs=[input_text], outputs=[audio])
    runBtn.click(fn=get_result, inputs=[inputs_video, inputs_image], outputs=[result])

    block.launch(share=True)

gr.mount_gradio_app(app, block, path='/')