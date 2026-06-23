# 🤖 Aiva — AI Virtual Assistant Chatbot

A fully custom-built AI voice and text chatbot — **no LLM API dependency**. Aiva uses a hand-trained **LSTM neural network** with a complete NLP pipeline (tokenization → lemmatization → word embeddings → intent classification) and responds via both text and synthesized speech through a custom desktop UI.

---

## ✨ Features

- 🎙️ **Voice Input** — real-time speech recognition via microphone (SpeechRecognition)
- 🔊 **Text-to-Speech Output** — Aiva speaks responses aloud using pyttsx3
- 🧠 **Custom LSTM Model** — trained from scratch with TensorFlow/Keras, no external AI API
- 📝 **Full NLP Pipeline** — NLTK tokenization + lemmatization + bag-of-words encoding
- 🎯 **Intent Classification** — 28+ intents covering real-world conversation scenarios
- 🖥️ **Modern Desktop UI** — built with customtkinter with audio wave animation
- 📊 **Training Visualization** — loss/accuracy plots with matplotlib & seaborn

---

## 🧠 How It Works

```
User Input (text / voice)
        ↓
  SpeechRecognition          ← converts voice to text
        ↓
  NLTK Tokenizer             ← splits sentence into words
        ↓
  NLTK Lemmatizer            ← reduces words to root form
        ↓
  Bag-of-Words Encoding      ← converts words to numeric vector
        ↓
  LSTM Neural Network        ← classifies intent (model.h5)
        ↓
  Response Selection         ← picks response from intents.json
        ↓
  pyttsx3 TTS Output         ← speaks the response aloud
```

---

## 🎯 Supported Intents (28+)

| Category | Examples |
|----------|---------|
| Conversation | Greeting, Goodbye, Thanks, Jokes |
| Orders | Order Status, Cancellation, Refund, Custom Orders |
| Business Info | Hours, Location, Contact, WiFi |
| Products | Items, Menu, Seasonal Products, Product Info, Allergies |
| Payments | Payment Methods, Gift Cards, Discounts |
| Customer | Loyalty Program, Membership, Feedback, Complaints |
| Business | Job Openings, Partnerships, Events, Sustainability |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow_Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge&logo=python&logoColor=white)

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | LSTM model training & inference |
| NLTK | Tokenization, lemmatization |
| scikit-learn | Data preprocessing, model evaluation |
| SpeechRecognition | Microphone voice input |
| pyttsx3 | Text-to-speech output |
| customtkinter | Modern desktop UI |
| Pillow | UI image handling |
| matplotlib / seaborn | Training metrics visualization |

---

## 📁 Project Structure

```
Aiva-Virtual-Assistant-Chatbot-Project/
├── intents.json        # Training data — 28+ intent patterns & responses
├── train_main.py       # NLP pipeline + LSTM model training
├── test_main.py        # Model evaluation & testing
├── main.py             # Main app — voice/text input, UI, response output
├── micCheck.py         # Microphone connectivity check
├── model.h5            # Trained Keras LSTM model
├── data.pickle         # Serialized words & intent classes
├── AI-UI.png           # UI assets
└── Audio_wave.png      # Audio animation asset
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- A working microphone (for voice input)

### Installation

```bash
# Clone the repository
git clone https://github.com/mbilalhussain15/Aiva-Virtual-Assistant-Chatbot-Project.git
cd Aiva-Virtual-Assistant-Chatbot-Project

# Create a virtual environment
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate

# Install dependencies
pip install nltk tensorflow scikit-learn speechrecognition matplotlib numpy pyttsx3 customtkinter pillow seaborn pywinstyles
```

### Step 1 — Train the Model

```bash
python train_main.py
```

This tokenizes and lemmatizes the intents, trains the LSTM network, and saves `model.h5` and `data.pickle`.

### Step 2 — Run Aiva

```bash
python main.py
```

Aiva's desktop UI will launch. Type or speak your query and Aiva will respond in text and voice.

### Optional — Test Microphone

```bash
python micCheck.py
```

---

## 📄 License

MIT © [Bilal Hussain](https://github.com/mbilalhussain15)
