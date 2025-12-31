# ai_friend.py
"""
Defensive Gradio AI Friend — corrected detection for messages mode.
If gr.ChatMessage exists, prefer messages format (dicts or ChatMessage).
"""

import json
import os
import time
import random
from typing import List, Any

import gradio as gr

# ---------- feature detection ----------
def supports_chatmessage() -> bool:
    return hasattr(gr, "ChatMessage")

def detect_messages_mode() -> bool:
    try:
        # harmless instantiate test (some gradio variants accept this)
        _ = gr.Chatbot(type="messages")
        return True
    except TypeError:
        return False
    except Exception:
        doc = getattr(gr.Chatbot, "__doc__", "") or ""
        return "messages" in doc

USE_CHATMESSAGE = supports_chatmessage()
# Prefer messages mode if ChatMessage exists OR detect_messages_mode says True
CHATBOT_USES_MESSAGES = USE_CHATMESSAGE or detect_messages_mode()

print(f"DEBUG: USE_CHATMESSAGE={USE_CHATMESSAGE}, CHATBOT_USES_MESSAGES={CHATBOT_USES_MESSAGES}")

# ---------- simple resources ----------
GROUNDING_SCRIPTS = [
    "5-4-3-2-1: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Breathe gently after each step.",
    "Deep breaths: Inhale for 4s, hold 4s, exhale 6s. Repeat 5 times, focusing on breath sensations.",
]
CRISIS_WORDS = {"panic", "suicide", "kill myself", "hurt myself", "can't breathe", "cant breathe", "help me", "crisis", "emergency"}

# ---------- persistence ----------
CHAT_HISTORY_FILE = "ai_friend_history.jsonl"
def append_history(entry: dict):
    try:
        with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print("WARNING: Failed to save history:", e)

# ---------- small helpers ----------
def contains_crisis(text: str) -> bool:
    lower = (text or "").lower()
    return any(w in lower for w in CRISIS_WORDS)

def make_supportive_reply(user_msg: str) -> str:
    if contains_crisis(user_msg):
        return ("I'm sorry — it sounds like you're in serious trouble. "
                "If you're in immediate danger please contact local emergency services.")
    if "ground" in (user_msg or "").lower():
        return GROUNDING_SCRIPTS[0]
    if "breath" in (user_msg or "").lower():
        return GROUNDING_SCRIPTS[1]
    return random.choice([
        "I'm sorry you're feeling this way. Would you like a short grounding exercise now? (type 'ground' or 'breath')",
        "Thanks for sharing. I can offer a short grounding exercise or breathing practice. Which would you like?"
    ])

# ---------- message normalization ----------
def make_message_obj(role: str, content: str):
    """
    Return:
     - gr.ChatMessage(...) if available & we are in messages mode and ChatMessage works
     - {"role": role, "content": content} when in messages mode (preferred)
     - (role, content) when in tuple mode
    """
    role_clean = (role or "user").strip().lower()
    if role_clean not in ("user", "assistant", "system"):
        role_clean = "user"
    content_str = "" if content is None else str(content)

    if CHATBOT_USES_MESSAGES:
        # Prefer ChatMessage dataclass if available, but fall back to dict if construction fails
        if USE_CHATMESSAGE:
            try:
                msg = gr.ChatMessage(role=role_clean, content=content_str)
                print("DEBUG: returning gr.ChatMessage:", msg)
                return msg
            except Exception as e:
                print("DEBUG: gr.ChatMessage constructor failed, falling back to dict:", e)
        msg = {"role": role_clean, "content": content_str}
        print("DEBUG: returning dict message:", msg)
        return msg
    else:
        t = (role_clean, content_str)
        print("DEBUG: returning tuple message:", t)
        return t

def history_to_chatbot(history: List[dict]):
    """
    Convert an internal list of dicts to UI format (dicts/ChatMessage when messages mode,
    tuples otherwise).
    """
    out = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            out.append(make_message_obj(item["role"], item["content"]))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append(make_message_obj(item[0], item[1]))
        else:
            print("DEBUG: skipping unknown history item:", item)
            continue
    print("DEBUG: final chatbot payload:", out)
    return out

# ---------- handlers ----------
def chat_handler(message, chat_history):
    # ensure chat_history is list[dict]
    if chat_history is None:
        chat_history = []
    if not isinstance(chat_history, list):
        try:
            chat_history = list(chat_history)
        except Exception:
            chat_history = []

    user_text = message or ""
    # append internal history
    chat_history.append({"role": "user", "content": user_text})

    reply = make_supportive_reply(user_text)
    chat_history.append({"role": "assistant", "content": reply})

    # persist to disk
    append_history({"role":"user","content":user_text})
    append_history({"role":"assistant","content":reply})

    ui_payload = history_to_chatbot(chat_history)

    # Must match outputs wired in UI: (chatbot, state, textbox)
    return ui_payload, chat_history, gr.update(value="")

def handle_upload(image, chat_history):
    if chat_history is None or not isinstance(chat_history, list):
        chat_history = []

    if image is None:
        chat_history.append({"role":"assistant","content":"No screenshot received. Please try again."})
        return history_to_chatbot(chat_history), chat_history

    os.makedirs("screenshots", exist_ok=True)
    path = f"screenshots/screenshot_{int(time.time())}.png"
    try:
        if hasattr(image, "save"):
            image.save(path)
        else:
            from PIL import Image
            import numpy as np
            arr = np.asarray(image)
            Image.fromarray(arr).save(path)
    except Exception as e:
        chat_history.append({"role":"assistant","content":f"Failed to save screenshot: {e}"})
        return history_to_chatbot(chat_history), chat_history

    chat_history.append({"role":"user","content":f"[screenshot saved: {path}]"})
    chat_history.append({"role":"assistant","content":"Got the screenshot — thank you."})
    append_history({"role":"user","content":f"[screenshot saved: {path}]"})
    return history_to_chatbot(chat_history), chat_history

# ---------- UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## YUVAi — AI Friend (Week 4)\nA supportive, rule-based chat. If you feel in immediate danger, contact emergency services.")

    # Create the Chatbot component. Keep CHATBOT_USES_MESSAGES as decided earlier.
    try:
        if CHATBOT_USES_MESSAGES:
            try:
                chatbot = gr.Chatbot(type="messages", label="Conversation")
            except Exception as e:
                # If the type kwarg isn't accepted by this Gradio build, create a plain Chatbot
                # but KEEP messages-mode True (we will return dicts/ChatMessage objects).
                print("WARNING: Chatbot(type='messages') not accepted; creating Chatbot() and keeping messages-mode. Reason:", e)
                chatbot = gr.Chatbot(label="Conversation")
        else:
            chatbot = gr.Chatbot(label="Conversation")
    except Exception as e:
        # Extremely conservative fallback to tuple mode only if creation fails
        print("WARNING: Failed to create Chatbot component; falling back to tuple mode:", e)
        # Do not change CHATBOT_USES_MESSAGES here if you want to preserve behavior elsewhere,
        # but if we truly cannot create a Chatbot we must fallback.
        chatbot = gr.Chatbot(label="Conversation")

    state = gr.State(value=[])
    txt = gr.Textbox(placeholder="Type how you're feeling...", lines=2)
    upload = gr.Image(type="pil", label="Upload screenshot after completing an exercise (optional)")

    with gr.Row():
        send_btn = gr.Button("Send")
        upload_btn = gr.Button("Upload Screenshot")

    # Note outputs must match returned values
    send_btn.click(fn=chat_handler, inputs=[txt, state], outputs=[chatbot, state, txt])
    upload_btn.click(fn=handle_upload, inputs=[upload, state], outputs=[chatbot, state])

    gr.Markdown("**Note:** This assistant is not a replacement for professional help. If you’re in immediate danger, contact local emergency services.")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
