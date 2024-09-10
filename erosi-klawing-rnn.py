# Tahap import library
import os
import rasterio
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Flatten, Input, Dropout, ConvLSTM2D
from tensorflow.keras.layers import Reshape
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow.keras.losses import Loss
import tensorflow.image as tfi
from keras import backend as K
from keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Menetapkan gambar ke folder path
IMAGE_FOLDER = '/content/drive/MyDrive/KlawingShape'

# Menetapkan sequence length dan batch size
SEQUENCE_LENGTH = 4
BATCH_SIZE = 2

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Mendapatkan list file gambar dari dalam folder
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith('.tif')]

# Ekstrak fitur pada gambar menggunakan ResNet50
features = []
for image_file in image_files:
    with rasterio.open(image_file) as src:
        img_array = src.read(1)
        img_array = np.stack((img_array,)*3, axis=-1)
        img_array = img_array.astype(np.uint8)
        img_array = np.array(Image.fromarray(img_array).resize((224, 224)))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img_features = base_model.predict(img_array)
        features.append(img_features)

# Reshape fitur
features = np.array(features)
features = features.reshape((len(image_files), 7, 7, 2048))

# Membuat sequences of features
X = []
y = []
for i in range(len(features) - SEQUENCE_LENGTH):
    X.append(features[i:i+SEQUENCE_LENGTH])
    y.append(features[i+SEQUENCE_LENGTH])
X = np.array(X)
y = np.array(y)

# Membagi data menjadi training dan validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Mendefinisikan K-fold cross-validation object
kf = KFold(n_splits=5)

# Inisialisasi lists untuk skor SSIM dan MSE
ssim_scores = []
mse_scores = []

#Melatih dan mengevaluasi model menggunakan cross validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Menyusun model
    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, 7, 7, 2048)))  
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(512, activation='tanh', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(LSTM(256, activation='tanh', return_sequences=False))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(7*7*2048, activation='sigmoid'))  
    model.add(Reshape((7, 7, 2048)))  

    # Mendefinisikan metrics
    def ssim_metric(y_true, y_pred):
        y_true_images = tf.reshape(y_true, (-1, 7, 7, 2048))  
        y_pred_images = tf.reshape(y_pred, (-1, 7, 7, 2048))  
        ssim_values = tf.map_fn(lambda x: 1 - tf.image.ssim(x[0], x[1], max_val=1.0, filter_size=3), (y_true_images, y_pred_images), dtype=tf.float32)
        return

    def mse_metric(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    # Compile model dengan metrics yang ada
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=ssim_metric,  
                  metrics=[ssim_metric, mse_metric])

    # Mendefinisikan early stop
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=5,  
        min_delta=0.001,  
        verbose=1  
    )

    # Melatih model menggunakan early stop
    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Menambahkan nilai MSE dan SSIM ke lists yang telah dibuat
    mse_scores.append(history.history['val_mse_metric'][-1])
    ssim_scores.append(history.history['val_ssim_metric'][-1])

# Print skor cross-validated SSIM dan MSE
print("Cross-validated MSE scores:", mse_scores)
print("Cross-validated SSIM scores:", ssim_scores)
print("Average MSE score:", np.mean(mse_scores))
print("Average SSIM score:", np.mean(ssim_scores))

# Membuat plot loss dan validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']

fig, ax = plt.subplots()
ax.plot(loss, label='Training Loss')
ax.plot(val_loss, label='Validation Loss')
ax.set_title('Loss and Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()