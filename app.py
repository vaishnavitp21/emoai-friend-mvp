# app.py
"""
Gradio voice -> Whisper transcription -> text emotion classification
Fallback to typed text if transcription fails.

Requirements (example):
pip install gradio transformers torch torchvision torchaudio
pip install -U openai-whisper      # or use faster-whisper for speed
# ffmpeg is required for whisper (install on system: e.g. apt, choco, brew)
"""

import os
import time
import logging
from pathlib import Path

import gradio as gr

# Whisper and transformers imports
try:
    import whisper             # openai-whisper package
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ---- Logging ----
logging.basicConfig(filename="app.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# ---- Models setup (loads once at startup) ----
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"  # compact emotion model on HF

def load_text_emotion_pipeline(model_name=MODEL_NAME):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        text_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
        return text_pipe
    except Exception as e:
        logging.exception("Failed to load emotion pipeline")
        raise

text_emotion_pipe = load_text_emotion_pipeline()

# Whisper model load (optional)
WHISPER_MODEL = "small"   # change to "tiny" or "small" depending on CPU speed
whisper_model = None
if WHISPER_AVAILABLE:
    try:
        whisper_model = whisper.load_model(WHISPER_MODEL)
    except Exception:
        whisper_model = None

# ---- Helpers ----
OUTPUT_FOLDER = Path("recordings")
OUTPUT_FOLDER.mkdir(exist_ok=True)

def save_audio_file(audio, prefix="audio"):
    """
    Gradio gives audio either as (sample_rate, np-array) or a temporary file path.
    This helper saves it to a wav file and returns file path.
    """
    # If audio is a file path (gr.Audio sometimes returns a temp file path)
    if isinstance(audio, str) and os.path.isfile(audio):
        dest = OUTPUT_FOLDER / f"{prefix}_{int(time.time())}.wav"
        os.replace(audio, dest)
        return str(dest)

    # If audio is (sr, np.ndarray)
    import soundfile as sf
    sr, data = audio
    dest = OUTPUT_FOLDER / f"{prefix}_{int(time.time())}.wav"
    sf.write(str(dest), data, sr)
    return str(dest)

def transcribe_with_whisper(audio_path):
    if whisper_model is None:
        raise RuntimeError("Whisper model not available on this machine.")
    # Use whisper to transcribe
    result = whisper_model.transcribe(audio_path)
    text = result.get("text", "").strip()
    return text

def classify_emotion_from_text(text: str):
    """
    Returns a sorted list of (label, score) and a top label.
    """
    if not text or text.strip() == "":
        return {"error": "No text provided for emotion classification."}
    try:
        preds = text_emotion_pipe(text[:1000])  # truncate to first 1000 chars for speed
        # preds is list of dicts with label and score for each candidate label
        # transform to sorted list
        scores = sorted(preds[0], key=lambda x: x["score"], reverse=True)
        top = scores[0]
        return {
            "text": text,
            "top_label": top["label"],
            "top_score": float(top["score"]),
            "all": [(d["label"], float(d["score"])) for d in scores]
        }
    except Exception:
        logging.exception("Emotion classification failed")
        return {"error": "Emotion classification failed (see app.log)."}

# ---- Gradio handlers ----
def handle_audio_submit(audio):
    """
    audio: file path or (sr, np-array) depending on gradio
    """
    try:
        if audio is None:
            return {"error": "No audio received. Please record or type text."}
        saved_path = save_audio_file(audio)
        logging.info(f"Saved audio to {saved_path}")

        # Transcribe
        try:
            transcription = transcribe_with_whisper(saved_path)
        except Exception as e:
            logging.exception("Transcription failed")
            # return error but allow typed-text fallback in UI
            return {"error": "Transcription failed. Please type text or check app.log."}

        # Classify
        result = classify_emotion_from_text(transcription)
        return result

    except Exception as e:
        logging.exception("handle_audio_submit failed")
        return {"error": "Unexpected error (see app.log)."}

def handle_text_submit(text):
    try:
        return classify_emotion_from_text(text)
    except Exception:
        logging.exception("handle_text_submit failed")
        return {"error": "Unexpected error (see app.log)."}

# ---- Build Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("## AI Friend — Voice Emotion Classifier\n**Record voice** (microphone) or **type text**. The app transcribes audio with Whisper and classifies emotion from the text.")
    # Top banner for emergency/contact (we'll add Step 1 instructions to make this editable)
    emergency = gr.Markdown("**EMERGENCY:** If you are in crisis, call your local emergency services or contact a trusted person.")
    # Audio input
    with gr.Row():
        audio_in = gr.Audio(source="microphone", type="filepath", label="Record voice (click mic)", elem_id="voice_input")
        transcribe_btn = gr.Button("Transcribe & Classify from Audio")
    with gr.Row():
        text_in = gr.Textbox(label="Or paste/type text here (fallback)", placeholder="Type here if audio fails...", lines=4, interactive=True)
        text_btn = gr.Button("Classify Text")

    output_top = gr.Textbox(label="Top emotion (label + score)", interactive=False)
    output_details = gr.JSON(label="Full scores / details")

    # Connect events
    transcribe_btn.click(fn=handle_audio_submit, inputs=[audio_in], outputs=[output_top, output_details],
                         show_progress=True)
    text_btn.click(fn=handle_text_submit, inputs=[text_in], outputs=[output_top, output_details])

    # small note about privacy
    gr.Markdown("**Privacy:** Audio is temporarily saved to `recordings/` on the server. Toggle 'Save Chat History' in your settings to persist transcripts.")

# Run
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=False)
