import pyaudio
import speech_recognition as sr


def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query.lower()
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return "Error: Timeout"
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return "Error: Unknown Value"
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return f"Error: {e}"

if __name__ == "__main__":
    take_command()

# List all available microphones
def list_microphones():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("Available microphones:")
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(f"Device {i}: {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
    p.terminate()

# Test the selected microphone
def test_microphone(device_index):
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=device_index)
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

if __name__ == "__main__":
    list_microphones()
    
    # Change the device_index to the correct microphone index from the list printed above
    device_index = int(input("Enter the device index you want to use: "))
    test_microphone(device_index)


