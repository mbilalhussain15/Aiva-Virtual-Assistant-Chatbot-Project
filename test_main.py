import os
import nltk
import numpy as np
import speech_recognition as sr
import pyttsx3
import customtkinter as ctk
from tkinter import PhotoImage, ttk, Toplevel
from tkinter import scrolledtext
from threading import Thread, Event
import pickle
import json
from tensorflow.keras.models import load_model
from nltk.stem.lancaster import LancasterStemmer
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pywinstyles

stemmer = LancasterStemmer()

# Initialize the log
log = []

# Load intents and model data
with open("intents.json") as file:
    data = json.load(file)

# Determine which model to use based on the dataset size
SMALL_DATASET_THRESHOLD = 200
num_patterns = sum(len(intent["patterns"]) for intent in data["intents"])

use_svm = num_patterns < SMALL_DATASET_THRESHOLD

if not use_svm:
    # Only load the LSTM model related files if necessary
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

    # Debug print for the shape of training data
    print(f"Shape of training data: {training.shape}")

    # Load trained LSTM model
    model = load_model("model.h5")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set English voice
voices = engine.getProperty('voices')
english_voice_set = False

for voice in voices:
    if hasattr(voice, 'languages') and len(voice.languages) > 0 and "en" in voice.languages[0]:
        engine.setProperty('voice', voice.id)
        english_voice_set = True
        break

if not english_voice_set:
    for voice in voices:
        if "English" in voice.name:
            engine.setProperty('voice', voice.id)
            break

# Load pre-trained SVM model and validation sets
svm_model = None
svm_model_path = "svm_model.pkl"
if use_svm:
    if os.path.exists(svm_model_path):
        with open(svm_model_path, "rb") as f:
            svm_model = pickle.load(f)
    else:
        raise FileNotFoundError(f"The SVM model file {svm_model_path} was not found.")

    with open("train_val_sets.pkl", "rb") as f:
        X_train_glove, X_val_glove, y_train_glove, y_val_glove = pickle.load(f)

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Define the relative path to the dataset and GloVe file
script_dir = os.path.dirname(os.path.abspath(__file__))
glove_file_path = os.path.join(script_dir, 'glove', 'glove.6B.50d.txt') 
embeddings_index = load_glove_embeddings(glove_file_path)

print(f"Loaded {len(embeddings_index)} word vectors.")

# Function to create bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag).reshape(1, 1, len(words))

# Function to get GloVe embedding
def get_glove_embedding(sentence, embeddings_index, embedding_dim=50):
    words = nltk.word_tokenize(sentence)
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    embedding_matrix = np.zeros((len(words), embedding_dim))
    for i, word in enumerate(words):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return np.mean(embedding_matrix, axis=0)

# Prepare data for SVM model using GloVe embeddings
X_glove = []
y_glove = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        embedding = get_glove_embedding(pattern, embeddings_index, embedding_dim=50)
        X_glove.append(embedding)
        y_glove.append(intent["tag"])

X_glove = np.array(X_glove)
y_glove = np.array(y_glove)

label_encoder = LabelEncoder()
y_glove = label_encoder.fit_transform(y_glove)

# Function to chat with the bot
def chat(query):
    if use_svm:
        embedding = get_glove_embedding(query, embeddings_index, embedding_dim=50)
        results = svm_model.predict([embedding])
        tag = label_encoder.inverse_transform(results)[0]
    else:
        results = model.predict(bag_of_words(query, words))
        results_index = np.argmax(results)
        tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    response = random.choice(responses)
    return response

# Function to update the log in the UI
def update_log(message):
    log_text.insert(ctk.END, message + '\n')
    log_text.see(ctk.END)
    print(message)

# Function to process voice commands
def process_command(query):
    update_log(f"User: {query}")
    if query == "quit":
        say("Goodbye!")
        stop_listening(log_text, stop_event)
    else:
        response = chat(query)
        update_log(f"Aiva: {response}")
        say(response)

# Function to speak the text
def say(text):
    engine.say(text)
    engine.runAndWait()
    log.append(f"Aiva: {text}")
    update_log(f"Aiva: {text}")

# Function to take command from the user
def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        update_log("Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            update_log("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            update_log(f"User said: {query}")
            return query.lower()
        except sr.WaitTimeoutError:
            update_log("Listening timed out while waiting for phrase to start")
            return "Some error occurred. Sorry from Aiva"
        except sr.UnknownValueError:
            update_log("Could not understand the audio")
            return "Some error occurred. Sorry from Aiva"
        except sr.RequestError as e:
            update_log(f"Could not request results; {e}")
            return "Some error occurred. Sorry from Aiva"
        except Exception as e:
            update_log(f"Error: {str(e)}")
            return "Some error occurred. Sorry from Aiva"

# Function to start listening for voice commands
def start_listening(log_text, text_entry, app, stop_event):
    stop_event.clear()
    def listen():
        while not stop_event.is_set():
            query = take_command()
            if query:
                process_command(query)
    listener_thread = Thread(target=listen)
    listener_thread.start()

# Function to stop listening for voice commands
def stop_listening(log_text, stop_event):
    stop_event.set()
    say("Goodbye!")

# Function to submit text command
def submit_text_command(log_text, text_entry, app):
    query = text_entry.get().strip()
    if query:
        log.append(f"User: {query}")
        update_log(f"User: {query}")
        process_command(query.lower())

# Function to create and display the LSTM heatmap
def show_lstm_heatmap():
    y_true = []
    y_pred = []

    # Reshape training data for prediction
    reshaped_training = training.reshape((training.shape[0], 1, training.shape[1]))

    # Generate predictions for the test set
    for i in range(len(reshaped_training)):
        results = model.predict(reshaped_training[i].reshape(1, 1, reshaped_training.shape[2]))
        results_index = np.argmax(results)
        y_pred.append(results_index)
        y_true.append(np.argmax(output[i]))

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='coolwarm')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('LSTM Confusion Matrix Heatmap')
    plt.show()

# Function to show SVM+GloVe heatmap
def show_svm_heatmap():
    y_pred_glove = svm_model.predict(X_val_glove)
    y_true_glove = y_val_glove

    # Create confusion matrix
    cm = confusion_matrix(y_true_glove, y_pred_glove)

    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='coolwarm')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('SVM+GloVe Confusion Matrix Heatmap')
    plt.show()

# Function to show LSTM model accuracy
def show_lstm_accuracy():
    reshaped_training = training.reshape((training.shape[0], 1, training.shape[1]))
    loss, accuracy = model.evaluate(reshaped_training, output, verbose=0)
    accuracy_percentage = accuracy * 100
    update_log(f"LSTM model accuracy: {accuracy_percentage:.2f}%")
    say(f"LSTM model accuracy is {accuracy_percentage:.2f} percent")
    show_accuracy_window("LSTM", accuracy_percentage)

# Function to show SVM+GloVe model accuracy
def show_svm_accuracy():
    y_pred_glove = svm_model.predict(X_val_glove)
    accuracy_glove = accuracy_score(y_val_glove, y_pred_glove)
    accuracy_percentage = accuracy_glove * 100
    update_log(f"SVM+GloVe model accuracy: {accuracy_percentage:.2f}%")
    say(f"SVM plus GloVe model accuracy is {accuracy_percentage:.2f} percent")
    show_accuracy_window("SVM+GloVe", accuracy_percentage)

# Function to display accuracy in a new window
def show_accuracy_window(model_name, accuracy_percentage):
    accuracy_window = Toplevel(app)
    accuracy_window.title(f"{model_name} Model Accuracy")
    accuracy_window.geometry("500x100")
    accuracy_label = ctk.CTkLabel(accuracy_window, text=f"{model_name} Model Accuracy: {accuracy_percentage:.2f}%", font=("Arial", 14), text_color="black")
    accuracy_label.pack(pady=20)

# Function to resize background image
def resize_bg(event):
    global background_photo
    new_width = event.width
    new_height = event.height
    background_img = Image.open("bg.png")
    background_img = background_img.resize((new_width, new_height), Image.LANCZOS)
    background_photo = ImageTk.PhotoImage(background_img)
    background_label.configure(image=background_photo)
    background_label.image = background_photo

# Create the Tkinter UI
def create_ui():
    global app, background_photo, background_label, log_text, text_entry, canvas, quit_heatmap_button

    app = ctk.CTk()  
    app.title("Aiva")
    app.geometry("1920x1080")  
    app.resizable(True, True)

    # Load and set the initial background image
    background_img = Image.open("bg.png")
    background_img = background_img.resize((1920, 1080), Image.LANCZOS)
    background_photo = ImageTk.PhotoImage(background_img)

    # Create a canvas for the background image
    background_label = ctk.CTkLabel(app, image=background_photo, text="")
    background_label.place(relx=0.5, rely=0.5, anchor="center")

    app.bind("<Configure>", resize_bg)  # Bind the resize event to the resize_bg function

    # Title label
    title_label = ctk.CTkLabel(app, text="Aiva", font=("Arial", 90, "bold"), text_color="white", bg_color='#434343')
    title_label.place(relx=0.45, rely=0.1, anchor="w")  
    pywinstyles.set_opacity(title_label, color="#434343")

    # Audio wave image
    audio_wave_img = Image.open("Audio_wave.png")
    audio_wave_img = audio_wave_img.resize((270, 270), Image.LANCZOS)
    audio_wave_photo = ImageTk.PhotoImage(audio_wave_img)
    audio_wave_label = ctk.CTkLabel(app, image=audio_wave_photo, text="", bg_color='#434343')
    audio_wave_label.place(relx=0.19, rely=0.43, anchor="w")

    pywinstyles.set_opacity(audio_wave_label, color="#434343")

    # Start Listening button
    start_button = ctk.CTkButton(app, text="Start Listening", command=lambda: start_listening(log_text, text_entry, app, stop_event), fg_color="transparent", border_color='green', border_width=1, corner_radius=7, hover_color="darkgreen")
    start_button.place(relx=0.05, rely=0.65, anchor="w")  
    # Stop Listening button
    stop_button = ctk.CTkButton(app, text="Stop Listening", command=lambda: stop_listening(log_text, stop_event), fg_color="transparent", border_color='red', border_width=1, corner_radius=7, hover_color="darkred")
    stop_button.place(relx=0.5, rely=0.65, anchor="e") 

    # Log text area
    log_text = ctk.CTkTextbox(app, width=270, height=300, wrap="word", fg_color="#333333", text_color="white")
    log_text.insert("1.0", "Welcome to the Voice Assistant")
    log_text.place(relx=0.75, rely=0.35, anchor="center")
    app.log_text = log_text 

    # Text entry for commands
    text_entry = ctk.CTkEntry(app, width=270, height=50, placeholder_text="Type text here...", fg_color="#333333", text_color="white")
    text_entry.place(relx=0.75, rely=0.55, anchor="center")
    app.text_entry = text_entry  

    # Submit command button
    submit_button = ctk.CTkButton(app, text="Submit Command", command=lambda: submit_text_command(log_text, text_entry, app), fg_color="#1E90FF", hover_color="#1C86EE")
    submit_button.place(relx=0.75, rely=0.65, anchor="center")

    # Quit button
    quit_button = ctk.CTkButton(app, text="Quit", command=app.quit, width=70, height=40, text_color="white", fg_color="transparent", border_color='red', border_width=1, corner_radius=7, hover_color="darkred")
    quit_button.place(relx=0.95, rely=0.95, anchor="center")

    if use_svm:
        # Add button to show SVM+GloVe heatmap
        svm_heatmap_button = ctk.CTkButton(app, text="Show SVM+GloVe Heatmap", command=show_svm_heatmap, fg_color="#FF5733", text_color="white")
        svm_heatmap_button.place(relx=0.05, rely=0.75, anchor="w")

        # Add button to show SVM+GloVe model accuracy
        svm_accuracy_button = ctk.CTkButton(app, text="Show SVM+GloVe Accuracy", command=show_svm_accuracy, fg_color="#4CAF50", text_color="white")
        svm_accuracy_button.place(relx=0.5, rely=0.75, anchor="w")
    else:
        # Add button to show LSTM heatmap
        heatmap_button = ctk.CTkButton(app, text="Show Heatmap", command=show_lstm_heatmap, fg_color="#FF5733", text_color="white")
        heatmap_button.place(relx=0.05, rely=0.75, anchor="w")

        # Add button to show LSTM model accuracy
        lstm_accuracy_button = ctk.CTkButton(app, text="Show Accuracy", command=show_lstm_accuracy, fg_color="#4CAF50", text_color="white")
        lstm_accuracy_button.place(relx=0.5, rely=0.75, anchor="e")

    return app

if __name__ == "__main__":
    stop_event = Event()
    app = create_ui()
    app.mainloop()
