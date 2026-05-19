# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# import os
# os.environ["TF_USE_LEGACY_KERAS"] = "1"

# # ==============================
# # Load artifacts
# # ==============================

# @st.cache_resource
# def load_artifacts():
#     model = load_model("model.keras")
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#     with open("config.pkl", "rb") as f:
#         config = pickle.load(f)
#     return model, tokenizer, config["max_len"]

# model, tokenizer, max_len = load_artifacts()

# # ==============================
# # Response function
# # ==============================

# def generate_response(text):
#     seq = tokenizer.texts_to_sequences([text.lower()])
#     seq = pad_sequences(seq, maxlen=max_len, padding='post')

#     pred = model.predict(seq, verbose=0)
#     pred = np.argmax(pred, axis=-1)[0]

#     index_to_word = {v: k for k, v in tokenizer.word_index.items()}

#     result = []
#     for idx in pred:
#         if idx != 0:
#             word = index_to_word.get(idx, "")
#             if word:
#                 result.append(word)

#     response = " ".join(result)
#     return response if response.strip() else "Sorry, I don't understand that question."

# # ==============================
# # Streamlit UI
# # ==============================

# st.set_page_config(page_title="Cambodia Tourism Chatbot", page_icon="🏛️")

# st.title("🏛️ Cambodia Tourism Chatbot")
# st.caption("Ask me about places, food, transport, or travel tips in Cambodia!")

# # Initialize chat history
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Display chat history
# st.subheader("Conversation")
# for speaker, text in st.session_state.history:
#     if speaker == "You":
#         st.write(f"🧑 **You:** {text}")
#     else:
#         st.write(f"🤖 **Bot:** {text}")

# # Input box
# user_input = st.text_input("Type your question:", key="input",
#                            placeholder="e.g. Where is Angkor Wat?")

# col1, col2, col3 = st.columns(3)

# with col1:
#     if st.button("Send ▶"):
#         if user_input.strip():
#             bot_response = generate_response(user_input)
#             st.session_state.history.append(("You", user_input))
#             st.session_state.history.append(("Bot", bot_response))
#             st.rerun()

# with col2:
#     if st.button("🧹 Clear Chat"):
#         st.session_state.history = []
#         st.rerun()

# with col3:
#     if st.button("❌ End Chat"):
#         st.session_state.history = []
#         st.info("Chat ended. Refresh to start again.")

# # ==============================
# # Sidebar: example questions
# # ==============================

# st.sidebar.title("💡 Try asking:")
# examples = [
#     "Where is Angkor Wat?",
#     "What food should I try?",
#     "Best time to visit Cambodia?",
#     "Is Cambodia safe?",
#     "How to travel in Phnom Penh?",
#     "What is Pub Street?",
#     "Where is Bayon Temple?",
# ]
# for ex in examples:
#     st.sidebar.markdown(f"- *{ex}*")
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================
# Load artifacts
# ==============================

## model = load_model("model.h5")
model = load_model("model.h5", m_format="keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]


# ==============================
# Response function
# ==============================

def generate_response(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')

    pred = model.predict(seq, verbose=0)
    pred = np.argmax(pred, axis=-1)[0]

    result = []
    for idx in pred:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                result.append(word)
                break

    return " ".join(result)


# ==============================
# Streamlit UI
# ==============================

st.title("SimpleRNN Chatbot 🤖")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
user_input = st.text_input("You:", key="input")

# Send button
if st.button("Send"):
    if user_input.strip() != "":
        bot_response = generate_response(user_input)

        # Save to history
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", bot_response))

# Display chat history
st.subheader("Conversation")

for speaker, text in st.session_state.history:
    if speaker == "You":
        st.write(f"🧑 You: {text}")
    else:
        st.write(f"🤖 Bot: {text}")

# ==============================
# End / Clear Chat
# ==============================

col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []

with col2:
    if st.button("❌ End Chat"):
        st.session_state.history = []
        st.write("Chat ended. Refresh to start again.")