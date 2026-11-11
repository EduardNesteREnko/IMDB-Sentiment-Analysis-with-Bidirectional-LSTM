# IMDB Sentiment Analysis with Bidirectional LSTM

This repository contains a deep learning pipeline implemented in TensorFlow/Keras for binary sentiment classification on the IMDB movie reviews dataset using a **Bidirectional LSTM** architecture.

## 📋 Overview of the Pipeline

1. **Dataset Loading**:
   * Loads the built-in IMDB dataset from Keras with a vocabulary size restriction (`max_features = 20000`).

2. **Text Processing & Padding**:
   * Prepares helper functions for sentence token splitting via regular expressions.
   * Standardizes sequence lengths to a fixed maximum length (`maxlen_sentence = 40`) using post-padding and post-truncation.

3. **Neural Network Architecture**:
   * **Embedding Layer**: Projects integer-encoded vocabulary indices into dense vectors of dimension 128.
   * **Stacked Bidirectional LSTMs**: Features two layers of bidirectional Long Short-Term Memory units (32 and 16 units respectively) to capture sequential dependencies in both directions.
   * **Dense & Dropout Layers**: Fully connected layers with ReLU activation and Dropout ($p = 0.5$) for regularization.
   * **Output Layer**: Single neuron with sigmoid activation for binary classification (positive/negative sentiment).

4. **Model Compilation & Training**:
   * Compiled using the `adam` optimizer and `binary_crossentropy` loss.
   * Trains for 3 epochs with a batch size of 32, evaluating performance on the test split.

---

## 🛠️ Code Implementation

```python
from tensorflow import keras
from tensorflow.keras import layers
import re

# 1. Load data
max_features = 20000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)

# Split reviews into individual sentences using punctuation marks
def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# Set sequence length and apply padding/truncating
maxlen_sentence = 40
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen_sentence, padding="post", truncating="post")
x_test = keras.utils.pad_sequences(x_test, maxlen=maxlen_sentence, padding="post", truncating="post")

# Build Bidirectional LSTM model
inputs = layers.Input(shape=(None,), dtype="int32")
x = layers.Embedding(max_features, 128)(inputs)
x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(16))(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.summary()

# Compile model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train model
batch_size = 32
epochs = 3

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=2
)
```

---

## 📊 Features

* **Bidirectional Recurrent Context**: Captures contextual dependencies from past and future tokens simultaneously.
* **Controlled Sequence Lengths**: Ensures uniform input tensor dimensions via Keras padding utilities.