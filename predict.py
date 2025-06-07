import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import nltk
from nltk.corpus import stopwords
import tkinter as tk
from tkinter import messagebox
import pygame

# Load Tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load Trained Model
model = tf.keras.models.load_model("model.h5")

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Function to predict spam or ham
def predict_text(text):
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=50)
    prediction = model.predict(padded_sequence)[0][0]
    return "Spam" if prediction > 0.5 else "Ham"

# Play sound for spam
def play_spam_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("spam_alert.mp3")  # Ensure the file exists in the correct path
    pygame.mixer.music.play()

# GUI for user input
def open_gui():
    def on_submit():
        user_input = entry.get()
        result = predict_text(user_input)
        print(f"User Input: {user_input}")
        print(f"Prediction: {result}")
        messagebox.showinfo("Prediction Result", f"This message is: {result}")
        if result == "Spam":
            play_spam_sound()
    
    root = tk.Tk()
    root.title("Twitter Spam Detector")
    root.geometry("400x200")
    
    label = tk.Label(root, text="Enter a tweet:")
    label.pack(pady=10)
    
    entry = tk.Entry(root, width=50)
    entry.pack(pady=5)
    
    button = tk.Button(root, text="Check", command=on_submit)
    button.pack(pady=10)
    
    root.mainloop()

# Run GUI
if __name__ == "__main__":
    open_gui()