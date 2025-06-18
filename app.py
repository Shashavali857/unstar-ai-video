# app.py (Pro Level - HD AI Video Generator with AI Voice + Text Watermark Only)

import gradio as gr
import torch
from diffusers import DiffusionPipeline
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, TextClip
from moviepy.audio.AudioClip import CompositeAudioClip
from gtts import gTTS
from deep_translator import GoogleTranslator
import os
import uuid

# Load zeroscope HD model
pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16
).to("cuda")

def translate_prompt(prompt):
    return GoogleTranslator(source='auto', target='en').translate(prompt)

def generate_video(prompt):
    session_id = str(uuid.uuid4())
    translated = translate_prompt(prompt)
    video_paths = []

    # Generate 5 short clips
    for i in range(5):
        frames = pipe(translated, num_inference_steps=40, guidance_scale=10).frames
        clip_path = f"{session_id}_scene{i}.mp4"
        frames[0].save(clip_path, format="mp4", save_all=True)
        video_paths.append(clip_path)

    # Join all 5 clips
    clips = [VideoFileClip(p).subclip(0, 6) for p in video_paths]
    stitched_video = concatenate_videoclips(clips)

    # Generate AI voice narration
    tts = gTTS(text=translated, lang='en')
    audio_path = f"{session_id}_audio.mp3"
    tts.save(audio_path)
    ai_voice = AudioFileClip(audio_path).subclip(0, stitched_video.duration)

    # Set only AI voice (no background music)
    final_video = stitched_video.set_audio(ai_voice)

    # Add "Unstar AI" text watermark at bottom right
    txt_clip = TextClip("Unstar AI", fontsize=30, color='white', font="Arial-Bold")
    txt_clip = txt_clip.set_duration(final_video.duration).set_pos(("right", "bottom")).margin(right=15, bottom=10)
    
    final = CompositeVideoClip([final_video, txt_clip])

    # Export video
    output_path = f"{session_id}_final_video.mp4"
    final.write_videofile(output_path, codec='libx264', audio_codec='aac')

    return output_path

# Gradio UI
gr.Interface(
    fn=generate_video,
    inputs=gr.Textbox(label="🎬 Describe Your Video (Hindi/English)", placeholder="e.g. Ek hero jungle me daud raha hai"),
    outputs=gr.Video(label="🎥 30s AI Video with Sound & Watermark"),
    title="🎞️ Unstar AI - HD Video Generator",
    description="Describe any scene in Hindi/English. This AI tool generates a 30s cinematic video with AI voice and a 'Unstar AI' text watermark. 100% Free & 24x7 on Render."
).launch(server_name="0.0.0.0", server_port=8000)
