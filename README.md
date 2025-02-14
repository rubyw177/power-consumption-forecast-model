# GRU Model for Time Series Forecasting

## Overview
This project implements a Gated Recurrent Unit (GRU) model for time series forecasting using TensorFlow and Keras. The notebook loads and preprocesses time series data, constructs a GRU-based neural network, and trains it with appropriate loss functions and optimization techniques.

## Features
- Uses TensorFlow and Keras for deep learning implementation
- Prepares time series data using windowing techniques
- Constructs a sequential GRU-based model for forecasting
- Implements EarlyStopping to prevent overfitting
- Uses Huber loss and Adam optimizer for training
- Visualizes loss curves for performance evaluation

## Dependencies
To run this notebook, install the following Python packages:

```bash
pip install tensorflow pandas numpy matplotlib
```

## Usage
1. Import the necessary libraries:
   ```python
   import tensorflow as tf
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.callbacks import EarlyStopping
   from tensorflow.keras.losses import Huber
   from tensorflow.keras.optimizers import Adam
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   ```
2. Define the function for time series windowing:
   ```python
   def window(series, window_size, batch_size, shuffle_buffer):
       series = tf.expand_dims(series, axis=-1)
       ds = tf.data.Dataset.from_tensor_slices(series)
       ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
       ```
3. Load and preprocess the time series data.
4. Define the GRU model architecture and compile it.
5. Train the model with EarlyStopping to optimize performance.
6. Visualize training loss and validation loss.

## Output
The model forecasts time series data based on trained parameters and provides loss curves to analyze performance.
