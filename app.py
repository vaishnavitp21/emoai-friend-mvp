# app.py
"""
Gradio voice -> Whisper transcription -> text emotion classification
Fallback to typed text if transcription fails.
"""

import os
import time
import json
import logging
from pathlib import Path

import gradio as gr

# ---------------- Whisper ----------------
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ---------------- Logging ----------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# ---------------- Models ----------------
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

def load_text_emotion_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )

text_emotion_pipe = load_text_emotion_pipeline()

WHISPER_MODEL = "small"
whisper_model = whisper.load_model(WHISPER_MODEL) if WHISPER_AVAILABLE else None

# ---------------- Audio helpers ----------------
OUTPUT_FOLDER = Path("recordings")
OUTPUT_FOLDER.mkdir(exist_ok=True)

def save_audio_file(audio):
    if isinstance(audio, str) and os.path.isfile(audio):
        dest = OUTPUT_FOLDER / f"audio_{int(time.time())}.wav"
        os.replace(audio, dest)
        return str(dest)

    import soundfile as sf
    sr, data = audio
    dest = OUTPUT_FOLDER / f"audio_{int(time.time())}.wav"
    sf.write(str(dest), data, sr)
    return str(dest)

def transcribe_audio(path):
    if whisper_model is None:
        raise RuntimeError("Whisper not available")
    result = whisper_model.transcribe(path)
    return result.get("text", "").strip()

def classify_emotion(text):
    if not text:
        return {"error": "No text provided"}

    preds = text_emotion_pipe(text[:1000])[0]
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)

    return {
        "text": text,
        "top_label": preds[0]["label"],
        "top_score": float(preds[0]["score"]),
        "all": [(p["label"], float(p["score"])) for p in preds]
    }

# ---------------- History saving ----------------
HISTORY_DIR = Path("history")
HISTORY_FILE = HISTORY_DIR / "emotions.json"
HISTORY_DIR.mkdir(exist_ok=True)

def load_history_safe():
    """
    Safely load emotion history.
    Returns empty list if file is missing, empty, or corrupted.
    """
    try:
        if not HISTORY_FILE.exists():
            return []

        content = HISTORY_FILE.read_text().strip()
        if not content:
            return []

        data = json.loads(content)
        if isinstance(data, list):
            return data

        return []

    except Exception:
        logging.exception("History file corrupted, resetting")
        return []

def save_emotion_if_allowed(result: dict, allow_save: bool):
    """
    Save emotion result safely if user allows it.
    - No raw audio
    - Redact long text
    """
    if not allow_save or "error" in result:
        return

    try:
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_label": result.get("top_label"),
            "top_score": result.get("top_score"),
            "text": (
                result.get("text", "")[:200] + "..."
                if len(result.get("text", "")) > 200
                else result.get("text", "")
            )
        }

        data = load_history_safe()
        data.append(entry)

        HISTORY_FILE.write_text(json.dumps(data, indent=2))

    except Exception:
        logging.exception("Failed to save emotion history")

# ---------------- Handlers ----------------
def handle_audio_submit(audio, allow_save):
    try:
        if audio is None:
            return "No audio provided", {}

        path = save_audio_file(audio)
        text = transcribe_audio(path)
        result = classify_emotion(text)

        save_emotion_if_allowed(result, allow_save)

        return f"{result['top_label']} ({result['top_score']:.2f})", result

    except Exception as e:
        logging.exception("Audio handler failed")
        return "Error", {"error": str(e)}

def handle_text_submit(text, allow_save):
    try:
        result = classify_emotion(text)
        save_emotion_if_allowed(result, allow_save)
        return f"{result['top_label']} ({result['top_score']:.2f})", result

    except Exception as e:
        logging.exception("Text handler failed")
        return "Error", {"error": str(e)}

# ---------------- UI ----------------
with gr.Blocks() as demo:

    gr.Markdown(
        """
⚠️ **Emergency Notice**  
If you are feeling unsafe or in crisis, please contact your local emergency number  
or reach out to a trusted person.  
This app is **not a medical or crisis service**.
"""
    )

    gr.Markdown(
        """
**Data & Privacy Notice**  
• Audio is processed locally for transcription  
• Transcripts are not saved unless enabled  
• You can turn this off anytime
"""
    )

    save_history = gr.Checkbox(label="Save Chat History", value=False)

    gr.Markdown("## AI Friend — Voice & Text Emotion Classifier")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Record voice"
        )
        audio_btn = gr.Button("Transcribe & Classify")

    with gr.Row():
        text_input = gr.Textbox(
            label="Or type text",
            placeholder="Type here if audio fails...",
            lines=4
        )
        text_btn = gr.Button("Classify Text")

    output_label = gr.Textbox(label="Top Emotion")
    output_json = gr.JSON(label="Full Emotion Scores")

    audio_btn.click(
        fn=handle_audio_submit,
        inputs=[audio_input, save_history],
        outputs=[output_label, output_json]
    )

    text_btn.click(
        fn=handle_text_submit,
        inputs=[text_input, save_history],
        outputs=[output_label, output_json]
    )

# ---------------- Run ----------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=False)
