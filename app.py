# app.py (Pro Level - HD AI Video Generator with Sound + Watermark)
import gradio as gr
import torch
from diffusers import DiffusionPipeline
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, ImageClip
from moviepy.audio.AudioClip import CompositeAudioClip
from gtts import gTTS
from deep_translator import GoogleTranslator
import os
import uuid

# Load Stable Diffusion video model (zeroscope)
pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16
).to("cuda")

# Translate prompt to English if in Hindi or any language
def translate_prompt(prompt):
    return GoogleTranslator(source='auto', target='en').translate(prompt)

# Generate cinematic video from prompt
def generate_video(prompt):
    session_id = str(uuid.uuid4())
    translated = translate_prompt(prompt)
    video_paths = []

    # Generate 5 short scenes
    for i in range(5):
        frames = pipe(translated, num_inference_steps=40, guidance_scale=10).frames
        clip_path = f"{session_id}_scene{i}.mp4"
        frames[0].save(clip_path, format="mp4", save_all=True)
        video_paths.append(clip_path)

    # Join scenes
    clips = [VideoFileClip(p).subclip(0, 6) for p in video_paths]
    stitched_video = concatenate_videoclips(clips)

    # AI voice (gTTS)
    tts = gTTS(text=translated, lang='en')
    audio_path = f"{session_id}_audio.mp3"
    tts.save(audio_path)
    ai_voice = AudioFileClip(audio_path).subclip(0, stitched_video.duration)

    # Optional background music
    if os.path.exists("background_music.mp3"):
        music = AudioFileClip("background_music.mp3").volumex(0.3)
        final_audio = CompositeAudioClip([ai_voice.volumex(1.0), music])
    else:
        final_audio = ai_voice

    stitched_video = stitched_video.set_audio(final_audio)

    # Add Unstar AI watermark
    if os.path.exists("unstar_logo.png"):
        logo = (ImageClip("unstar_logo.png")
                .set_duration(stitched_video.duration)
                .resize(height=60)
                .margin(right=10, bottom=10, opacity=0)
                .set_pos(("right", "bottom")))
        final = CompositeVideoClip([stitched_video, logo])
    else:
        final = stitched_video

    output_path = f"{session_id}_final_video.mp4"
    final.write_videofile(output_path, codec='libx264', audio_codec='aac')

    return output_path

# Gradio UI
gr.Interface(
    fn=generate_video,
    inputs=gr.Textbox(label="🎬 Describe Your Video (Hindi/English)", placeholder="e.g. Ek hero jungle me daud raha hai"),
    outputs=gr.Video(label="🎥 30s AI Video with Sound & Watermark"),
    title="🎞️ Unstar AI - HD Video Generator",
    description="Enter a prompt in Hindi or English. This tool will generate 5 scenes and auto merge into one 30s HD video with AI voice, optional music, and Unstar logo. 100% Free."
).launch()
