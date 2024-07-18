import os
import nltk
import pickle
import json
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import speech_recognition as sr
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Initialize the stemmer
stemmer = LancasterStemmer()

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

# Function to count the total number of patterns
def count_patterns(intents):
    return sum(len(intent["patterns"]) for intent in intents["intents"])

# Define threshold for small and large datasets
SMALL_DATASET_THRESHOLD = 200

# Count the number of patterns in the dataset
num_patterns = count_patterns(data)
print("NUM PATTERNS")
print(num_patterns)

if num_patterns < SMALL_DATASET_THRESHOLD:
    # Use SVM+GloVe model
    print("Using SVM+GloVe model")
    
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

    # Convert text to GloVe vectors
    def text_to_glove(text, embeddings_index, embedding_dim=50):
        words = text.split()
        vec = np.zeros(embedding_dim)
        count = 0
        for word in words:
            if word in embeddings_index:
                vec += embeddings_index[word]
                count += 1
        if count != 0:
            vec /= count
        return vec

    # Define the relative path to the dataset and GloVe file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    glove_file_path = os.path.join(script_dir, 'glove', 'glove.6B.50d.txt')  
    
    embeddings_index = load_glove_embeddings(glove_file_path)
    embedding_dim = 50  # Update to match the dimensions of your GloVe file
    print(f"Loaded {len(embeddings_index)} word vectors.")

    X_glove = []
    y_glove = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            embedding = text_to_glove(pattern, embeddings_index, embedding_dim)
            X_glove.append(embedding)
            y_glove.append(intent["tag"])

    X_glove = np.array(X_glove)
    y_glove = np.array(y_glove)

    label_encoder = LabelEncoder()
    y_glove = label_encoder.fit_transform(y_glove)

    # Split the data into training and validation sets
    X_train_glove, X_val_glove, y_train_glove, y_val_glove = train_test_split(X_glove, y_glove, test_size=0.2, random_state=42)

    # Save the training and validation sets
    with open("train_val_sets.pkl", "wb") as f:
        pickle.dump((X_train_glove, X_val_glove, y_train_glove, y_val_glove), f)

    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_glove, y_train_glove)

    # Save the SVM model
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(svm_model, f)

    # Evaluate the SVM model
    y_pred_glove = svm_model.predict(X_val_glove)
    accuracy_glove = accuracy_score(y_val_glove, y_pred_glove)
    print(f"SVM model accuracy: {accuracy_glove * 100:.2f}%")

else:
    # Use LSTM model
    print("Using LSTM model")

    # Load or preprocess data
    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        training = np.array(training)
        output = np.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    # Reshape training data for LSTM layer
    training = np.array(training)
    training = training.reshape((training.shape[0], 1, training.shape[1]))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(training, output, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, len(training[0][0])), return_sequences=True))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(64))
    model.add(Dense(len(output[0]), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train LSTM model
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val))
    model.save("model.h5")  # Save the entire LSTM model

    # Evaluate LSTM model
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"LSTM model accuracy: {accuracy * 100:.2f}%")

# Function to capture audio
def capture_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = r.listen(source, timeout=10)  # Added timeout to avoid infinite listening
            print("Recognizing...")
            signal = np.frombuffer(audio.get_raw_data(), dtype=np.int16)  # Convert audio data to numpy array
            sampling_rate = audio.sample_rate
            print(f"Audio captured. Sampling rate: {sampling_rate}, Signal length: {len(signal)}")
            return signal, sampling_rate
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None, None
        except sr.RequestError as e:
            print(f"Error: {str(e)}")
            return None, None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None, None

# Function for Fourier transform
def fourier_transform(signal, sampling_rate):
    print("Computing Fourier transform")
    N = len(signal)
    T = 1.0 / sampling_rate
    yf = fft(signal)
    xf = fftfreq(N, T)[:N//2]
    return xf, 2.0 / N * np.abs(yf[0:N//2])

# Function to compute and show Fourier transform
def compute_and_show_fourier():
    print("Starting audio capture for Fourier transform")
    signal, sampling_rate = capture_audio()
    if signal is None:
        print("Error capturing audio.")
        return
    
    print("Audio captured successfully, proceeding with Fourier transform")
    xf, yf = fourier_transform(signal, sampling_rate)

    # Plotting the Fourier Transform
    print("Plotting Fourier transform")
    plt.figure(figsize=(8, 6))
    plt.plot(xf, yf)
    plt.title('Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    print("Fourier transform plotted successfully")

# Example usage
compute_and_show_fourier()
