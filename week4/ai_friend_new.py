# ai_friend_fixed.py
import json
import os
import time
import random
from typing import List, Tuple, Any

import gradio as gr

# ---------- detect Chatbot message mode (works across Gradio versions) ----------
CHATBOT_USES_MESSAGES = False
try:
    # if the current gradio supports 'type="messages"' this will succeed
    _test_chatbot = gr.Chatbot(type="messages")
    CHATBOT_USES_MESSAGES = True
    del _test_chatbot
except TypeError:
    CHATBOT_USES_MESSAGES = False
except Exception:
    # best effort fallback
    CHATBOT_USES_MESSAGES = getattr(gr.Chatbot, "__call__", False) is not None and hasattr(gr, "Chatbot")

# ---------- sample grounding scripts / utilities ----------
GROUNDING_SCRIPTS = [
    "5-4-3-2-1: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Breathe gently after each step.",
    "Deep breaths: Inhale for 4s, hold 4s, exhale 6s. Repeat 5 times, focusing on breath sensations.",
    "Box breathing: Inhale 4, hold 4, exhale 4, hold 4. Repeat 4 cycles and notice calming.",
]

CRISIS_WORDS = {"panic", "help me", "suicide", "kill myself", "hurt myself", "overwhelmed", "can't breathe", "cant breathe", "crisis", "emergency"}

CHAT_HISTORY_FILE = "ai_friend_history.jsonl"


def append_history(entry: dict):
    try:
        with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Failed to save history:", e)


def contains_crisis(text: str) -> bool:
    lower = text.lower()
    for w in CRISIS_WORDS:
        if w in lower:
            return True
    return False


def make_supportive_reply(user_msg: str) -> str:
    if contains_crisis(user_msg):
        return "If you are in immediate danger please contact local emergency services now."
    lm = user_msg.lower()
    if "ground" in lm:
        return GROUNDING_SCRIPTS[0] + "\n\nWhen you finish, upload a screenshot or type 'done'."
    if "breath" in lm or "breathe" in lm:
        return GROUNDING_SCRIPTS[1] + "\n\nWhen you finish, upload a screenshot or type 'done'."
    templates = [
        "I'm sorry you're feeling this way. Would you like a grounding exercise or breathing practice? (type 'ground' or 'breath')",
        "Thanks for sharing — I can guide a short grounding or breathing exercise. Which would you prefer?"
    ]
    return random.choice(templates)


# ---------- sanitize helpers ----------
def _to_dict_message(item: Any) -> dict:
    """Convert tuple or dict-like item to a dict message with 'role' and 'content' keys."""
    if isinstance(item, dict):
        # prefer 'content' key; fall back to 'text'
        role = item.get("role") or item.get("from") or "assistant"
        content = item.get("content", item.get("text", ""))
        return {"role": role, "content": content}
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        # expected tuple (role, content)
        return {"role": item[0], "content": item[1]}
    # unknown type -> wrap into assistant message
    return {"role": "assistant", "content": str(item)}


def sanitize_history_for_messages(history: List[Any]) -> List[dict]:
    """Return a list of dicts each with 'role' and 'content' suitable for messages-mode Chatbot."""
    sanitized = []
    for i, it in enumerate(history):
        if isinstance(it, dict) and "role" in it and "content" in it:
            sanitized.append({"role": it["role"], "content": it["content"]})
        else:
            # convert tuple or other forms -> dict
            sanitized.append(_to_dict_message(it))
    return sanitized


def sanitize_history_for_tuples(history: List[Any]) -> List[tuple]:
    """Return a list of (role, content) tuples suitable for tuple-mode Chatbot."""
    sanitized = []
    for it in history:
        if isinstance(it, tuple) and len(it) >= 2:
            sanitized.append((it[0], it[1]))
        elif isinstance(it, dict):
            sanitized.append((it.get("role", "assistant"), it.get("content", it.get("text", ""))))
        else:
            sanitized.append(("assistant", str(it)))
    return sanitized


# ---------- convert history to Chatbot expected format ----------
def _history_to_chatbot_messages(history: List[dict]):
    """
    Robust conversion of internal history -> format accepted by gr.Chatbot.

    - If CHATBOT_USES_MESSAGES True -> return list[dict] with keys 'role' and 'content'
    - Else -> return list[tuple] (role, content)
    This routine will coerce common shapes (dicts with different keys, tuples,
    strings) and never return an invalid item.
    """
    out = []
    if history is None:
        history = []

    for i, h in enumerate(history):
        # dict-like (expected)
        if isinstance(h, dict):
            role = h.get("role") or h.get("type") or "user"
            content = h.get("content") or h.get("text") or h.get("message") or ""
            if CHATBOT_USES_MESSAGES:
                out.append({"role": role, "content": content})
            else:
                out.append((role, content))
            continue

        # tuple/list-like
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            role, content = h[0], h[1]
            # ensure role/content are strings
            role = str(role)
            content = str(content)
            if CHATBOT_USES_MESSAGES:
                out.append({"role": role, "content": content})
            else:
                out.append((role, content))
            continue

        # fallback: arbitrary object -> stringify as assistant message
        text = str(h)
        if CHATBOT_USES_MESSAGES:
            out.append({"role": "assistant", "content": text})
        else:
            out.append(("assistant", text))

    # TEMP DEBUG (comment out when done) — prints the shape sent to the Chatbot
    # print("DEBUG: _history_to_chatbot_messages ->", out)

    return out



# ---------- handlers ----------
def chat_handler(message, chat_history):
    if chat_history is None:
        chat_history = []

    # ensure internal history is list of dicts (we keep internal state as dicts)
    if not (isinstance(chat_history, list) and all(isinstance(x, dict) for x in chat_history)):
        # try to coerce
        chat_history = sanitize_history_for_messages(chat_history)

    _append_message(chat_history, "user", message)
    reply = make_supportive_reply(message)
    _append_message(chat_history, "assistant", reply)

    # persist disk log
    append_history({"role": "user", "content": message})
    append_history({"role": "assistant", "content": reply})

    # return the *format* expected by the Chatbot component and the internal state (dicts)
    return _history_to_chatbot_messages(chat_history), chat_history


def handle_upload(image, chat_history):
    if chat_history is None:
        chat_history = []

    if not (isinstance(chat_history, list) and all(isinstance(x, dict) for x in chat_history)):
        chat_history = sanitize_history_for_messages(chat_history)

    if image is None:
        _append_message(chat_history, "assistant", "No screenshot received. Please try again.")
        return _history_to_chatbot_messages(chat_history), chat_history

    os.makedirs("screenshots", exist_ok=True)
    save_path = f"screenshots/screenshot_{int(time.time())}.png"
    try:
        image.save(save_path)
    except Exception:
        # fallback: image might be numpy array
        try:
            from PIL import Image
            im = Image.fromarray(image)
            im.save(save_path)
        except Exception as e:
            _append_message(chat_history, "assistant", f"Failed to save screenshot: {e}")
            return _history_to_chatbot_messages(chat_history), chat_history

    append_history({"role": "user", "content": f"[screenshot saved: {save_path}]"})
    _append_message(chat_history, "assistant", "Got the screenshot — thank you. Type 'next' for another exercise or 'stop' to finish.")
    return _history_to_chatbot_messages(chat_history), chat_history


# ---------- UI ----------
with gr.Blocks() as demo:
    gr.Markdown("## YUVAi — AI Friend (Week 4)\nA supportive, rule-based chat. If you feel in immediate danger, contact emergency services.")

    # create Chatbot according to detected mode
    try:
        if CHATBOT_USES_MESSAGES:
            chatbot = gr.Chatbot(type="messages")
        else:
            chatbot = gr.Chatbot()
    except TypeError:
        CHATBOT_USES_MESSAGES = False
        chatbot = gr.Chatbot()

    # internal state for our app: list of dicts with keys 'role' and 'content'
    state = gr.State([])

    txt = gr.Textbox(placeholder="Type how you're feeling...", lines=2)
    upload = gr.Image(type="pil", label="Upload screenshot after completing an exercise (optional)")

    with gr.Row():
        send_btn = gr.Button("Send")
        upload_btn = gr.Button("Upload Screenshot")

    send_btn.click(fn=chat_handler, inputs=[txt, state], outputs=[chatbot, state])
    upload_btn.click(fn=handle_upload, inputs=[upload, state], outputs=[chatbot, state])

    gr.Markdown("**Note:** This assistant is not a replacement for professional help. If you’re in immediate danger, contact local emergency services.")


if __name__ == "__main__":
    # change the port if 7860 is in use
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
