##### Version 1
# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ==============================
# # Load artifacts
# # ==============================

# ## model = load_model("model.h5")
# model = load_model("model_v2.h5")
# with open("tokenizer_v2.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# with open("config_v2.pkl", "rb") as f:
#     config = pickle.load(f)

# max_len = config["max_len"]


# # ==============================
# # Response function
# # ==============================

# def generate_response(text):
#     seq = tokenizer.texts_to_sequences([text])
#     seq = pad_sequences(seq, maxlen=max_len, padding='post')

#     pred = model.predict(seq, verbose=0)
#     pred = np.argmax(pred, axis=-1)[0]

#     result = []
#     for idx in pred:
#         for word, index in tokenizer.word_index.items():
#             if index == idx:
#                 result.append(word)
#                 break

#     return " ".join(result)


# # ==============================
# # Streamlit UI
# # ==============================

# st.title("SimpleRNN Chatbot 🤖")

# # Initialize chat history
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Input box
# user_input = st.text_input("You:", key="input")

# # Send button
# if st.button("Send"):
#     if user_input.strip() != "":
#         bot_response = generate_response(user_input)

#         # Save to history
#         st.session_state.history.append(("You", user_input))
#         st.session_state.history.append(("Bot", bot_response))

# # Display chat history
# st.subheader("Conversation")

# for speaker, text in st.session_state.history:
#     if speaker == "You":
#         st.write(f"🧑 You: {text}")
#     else:
#         st.write(f"🤖 Bot: {text}")

# # ==============================
# # End / Clear Chat
# # ==============================

# col1, col2 = st.columns(2)

# with col1:
#     if st.button("🧹 Clear Chat"):
#         st.session_state.history = []

# with col2:
#     if st.button("❌ End Chat"):
#         st.session_state.history = []
#         st.write("Chat ended. Refresh to start again.")
##### Version 2
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cambodia Tourism Chatbot",
    page_icon="🇰🇭",
    layout="centered"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .user-bubble {
        background: #1a73e8;
        color: white;
        padding: 10px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0 6px 20%;
        font-size: 0.95rem;
    }
    .bot-bubble {
        background: #f1f3f4;
        color: #333;
        padding: 10px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 20% 6px 0;
        font-size: 0.95rem;
        border-left: 3px solid #e8a000;
    }
    .conf-tag {
        font-size: 0.72rem;
        color: #888;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model("model_v2.h5")
    with open("tokenizer_v2.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("config_v2.pkl", "rb") as f:
        config = pickle.load(f)
    return model, tokenizer, config["max_len"], config["id_to_answer"]

try:
    model, tokenizer, max_len, id_to_answer = load_artifacts()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.error(f"⚠️ Could not load model: {e}\n\nMake sure model.h5, tokenizer.pkl, and config.pkl are in the same folder.")

# ─── Response Function ────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25

def generate_response(text: str) -> tuple[str, float]:
    """
    Classify the input question and return (answer_text, confidence).

    The model outputs a probability vector over all answer classes.
    argmax picks the winning class, and id_to_answer retrieves the
    pre-written answer string for that class.
    """
    seq    = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    probs      = model.predict(padded, verbose=0)[0]   # shape: (num_classes,)
    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id])

    if confidence < CONFIDENCE_THRESHOLD:
        return (
            "I'm not sure about that. Try asking about Cambodia's temples, "
            "food, transport, visa, or beaches!",
            confidence
        )

    return id_to_answer[class_id], confidence

# ─── Session State ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []    # list of {"role": "user"|"bot", "text": str, "conf": float}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇰🇭 Cambodia Tourism Bot")
    st.markdown("Ask me anything about visiting Cambodia!")
    st.markdown("---")
    st.markdown("**💡 Try these questions:**")
    examples = [
        "Where is Angkor Wat?",
        "What food should I try?",
        "Best time to visit Cambodia?",
        "Is Cambodia safe?",
        "How to travel in Phnom Penh?",
        "What is Pub Street?",
        "Where is Bayon Temple?",
        "How do I get a visa?",
        "What currency does Cambodia use?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

    st.markdown("---")
    st.markdown(f"**Session messages:** {len(st.session_state.history)}")
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        st.rerun()

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.title("🇰🇭 Cambodia Tourism Chatbot")
st.caption("Powered by SimpleRNN · Ask me about places, food, transport, and travel tips!")


# Display conversation history
if not st.session_state.history:
    st.info("👋 Hello! Ask me anything about travelling to Cambodia.")
else:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-bubble'>🧑 {msg['text']}</div>",
                unsafe_allow_html=True
            )
        else:
            conf_pct = f"{msg['conf']:.0%}"
            st.markdown(
                f"<div class='bot-bubble'>🤖 {msg['text']}"
                f"<div class='conf-tag'>confidence: {conf_pct}</div></div>",
                unsafe_allow_html=True
            )

st.markdown("---")

# Input form — clear_on_submit resets the box after each send
with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input(
            label="Your message",
            placeholder="e.g. Where is Angkor Wat?",
            label_visibility="collapsed"
        )
    with col_btn:
        submitted = st.form_submit_button("Send")

if submitted and user_input.strip() and MODEL_OK:
    answer, conf = generate_response(user_input.strip())
    st.session_state.history.append({"role": "user", "text": user_input.strip(), "conf": 1.0})
    st.session_state.history.append({"role": "bot",  "text": answer})
    st.rerun()
