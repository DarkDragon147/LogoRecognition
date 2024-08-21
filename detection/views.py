from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.DEBUG)

# Create your views here.

def index(request):
    return render(request, 'detection/index.html')

def verify_model():
    # Load and compile the model within the request context
    model = load_model('fake_logo_detector.h5')
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Load a known test image
    img = image.load_img('Google-symbol.png', target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0

    # Make prediction using the loaded and compiled model
    prediction = model.predict(img_tensor)
    logging.debug(f"Verification Prediction: {prediction}")
    result = 'Fake' if prediction < 0.5 else 'Real'
    logging.debug(f"Verification Result: {result}")

def predict(request):
    if request.method == 'POST':
        logging.debug("POST request received")
        file = request.FILES.get('image')
        if not file:
            logging.error("No file uploaded")
            return render(request, 'detection/index.html', {'error': 'No file uploaded'})

        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        file_url = fs.url(filename)

        img_path = fs.path(filename)
        logging.debug(f"Image Path: {img_path}")

        try:
            img = image.load_img(img_path, target_size=(150, 150))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.0

            logging.debug(f"Image Tensor Shape: {img_tensor.shape}")

            model = load_model('fake_logo_detector.h5')
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            prediction = model.predict(img_tensor)
            logging.debug(f"Prediction: {prediction}")
            result = 'Fake' if prediction < 0.5 else 'Real'
            logging.debug(f"Result: {result}")

            return render(request, 'detection/result.html', {'result': result, 'file_url': file_url})
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return render(request, 'detection/index.html', {'error': 'Error processing image'})
    return render(request, 'detection/index.html')
