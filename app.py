# Set the environment variable to avoid OpenMP runtime error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import whisper
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip
import cv2

app = Flask(__name__, template_folder='te')
UPLOAD_FOLDER = 'uploaded_videos'
PROCESSED_FOLDER = 'processed_gifs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Load the Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded.")

def detect_silence(audio_path, silence_thresh=-40, min_silence_len=400):
    audio = AudioSegment.from_file(audio_path)
    silent_ranges = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    silent_ranges = [(start / 1000, end / 1000) for start, end in silent_ranges]
    return silent_ranges

def get_non_silent_segments(silent_segments, video_duration):
    non_silent_segments = []
    previous_end = 0
    for start, end in silent_segments:
        if start > previous_end:
            non_silent_segments.append((previous_end, start))
        previous_end = end
    if previous_end < video_duration:
        non_silent_segments.append((previous_end, video_duration))
    return non_silent_segments

def cut_video_segment(video_path, start_time, end_time, output_path):
    with VideoFileClip(video_path) as video:
        new_clip = video.subclip(start_time, end_time)
        new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

def convert_video_to_audio(video_path, audio_path):
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)
    video_clip.close()

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

def overlay_text_on_video(video_path, output_path, text):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        font_scale = 3
        thickness = 10
        color = (255, 255, 255)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = int((width - text_size[0]) / 2)
        text_y = height - 50
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()

def convert_video_to_gif(input_video_path, output_gif_path):
    clip = VideoFileClip(input_video_path)
    clip.write_gif(output_gif_path)

def process_video(video_path, output_dir):
    audio_path = "temp_audio.wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    
    silent_segments = detect_silence(audio_path)
    video_duration = video.duration
    non_silent_segments = get_non_silent_segments(silent_segments, video_duration)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_files = []
    for i, (start_time, end_time) in enumerate(non_silent_segments):
        video_segment_path = f"{output_dir}/segment_{i}.mp4"
        cut_video_segment(video_path, start_time, end_time, video_segment_path)
        
        segment_audio_path = f"{output_dir}/segment_{i}_audio.wav"
        convert_video_to_audio(video_segment_path, segment_audio_path)
        transcription = transcribe_audio(segment_audio_path)
        video_with_text_path = f"{output_dir}/segment_{i}_with_text.mp4"
        overlay_text_on_video(video_segment_path, video_with_text_path, transcription)
        
        gif_path = f"{output_dir}/segment_{i}.gif"
        convert_video_to_gif(video_with_text_path, gif_path)
        
        os.remove(video_segment_path)
        os.remove(segment_audio_path)
        os.remove(video_with_text_path)
        
        output_files.append(gif_path)
    
    os.remove(audio_path)
    return output_files

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        process_video(file_path, app.config['PROCESSED_FOLDER'])
        return redirect(url_for('processed_files'))

@app.route('/processed_files')
def processed_files():
    files = os.listdir(app.config['PROCESSED_FOLDER'])
    return render_template('processed_files.html', files=files)

@app.route('/processed_files/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
