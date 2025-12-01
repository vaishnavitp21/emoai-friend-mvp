import gradio as gr
from transformers import pipeline

# Load a simple emotion classifier (HuggingFace)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def predict_emotion(text):
    if not text.strip():
        return {"neutral": 1.0}

    results = emotion_classifier(text)[0]

    # Convert list of dicts → {label: score}
    output = {item["label"]: float(item["score"]) for item in results}

    return output


# Gradio UI
with gr.Blocks(title="YUVAi — Emotion Detection (Week 2)") as demo:
    gr.Markdown("## 💬 Text Emotion Detector\nEnter any text and click **Predict Emotion**.")

    input_box = gr.Textbox(lines=6, label="Enter text")
    output_label = gr.Label(label="Emotion Scores")

    predict_btn = gr.Button("Predict Emotion")
    predict_btn.click(fn=predict_emotion, inputs=input_box, outputs=output_label)

demo.launch()
