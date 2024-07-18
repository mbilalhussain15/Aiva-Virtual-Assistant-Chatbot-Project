# Aiva Virtual Assistant

Welcome to the Aiva Virtual Assistant repository! This project is designed to develop a voice assistant capable of understanding and responding to user queries. It includes two main Python scripts: `train_main.py` and `test_main.py`. The first script trains the model, and the second script tests the trained model.

## Getting Started

Follow the steps below to set up and run the project.

### Prerequisites

Ensure you have the following software installed on your system:
- Python 3.x

### Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-username/aiva-virtual-assistant.git
    cd aiva-virtual-assistant
    ```

2. Create a virtual environment (recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install nltk tensorflow scikit-learn speechrecognition matplotlib numpy pyttsx3 customtkinter pillow seaborn pywinstyles
    ```

## Running the Project

### Step 1: Train the Model

Run the `train_main.py` script to train your model.
```bash
python train_main.py
```

### Step 2: Test the Model

After successfully training the model, run the test_main.py script to test the trained model.

```bash
python test_main.py
```
