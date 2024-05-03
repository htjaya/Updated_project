from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import keras
import numpy as np
from flask_cors import CORS
from PIL import Image 
import io
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('./Model/trained_model.h5')

# Define a function to process the uploaded image
def process_image(img):
    img_stream = io.BytesIO(img.read())
    
    # Open the image using PIL (Pillow)
    img = Image.open(img_stream)
    
    # Resize the image
    img = img.resize((64, 64))
    
    # Convert the image to a numpy array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    return img_array

@app.route('/', methods=['GET'])
def index():
    return jsonify({'success': 'Smart Banana Hub Python Backend Root'})

# Define an endpoint for image upload
@app.route('/FileUpload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    img = process_image(file)
    
    predictions = model.predict(img)
    print("predictions result : ")
    print(predictions)
    print("-------------------------------")
    result = {'banana_probability': predictions[0]}  # Assuming second class is banana

    test_set = keras.utils.image_dataset_from_directory('./test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,

    )

    print("Test Class : ")
    print(test_set.class_names)
    print("-------------------------------")

    print("Prediction : ")
    print(predictions[0])
    print("-------------------------------")

    print("Max Prediction : ")
    print(max(predictions[0]))
    print("-------------------------------")

    result_index = np.where(predictions[0] == max(predictions[0]))
    print(result_index[0][0])

    print('It is a {}'.format(test_set.class_names[result_index[0][0]]))
    resultData = {
    'statusCode':200,
    'status':'Success',
    'message': 'Banana Found.', 
    'predictedResult': test_set.class_names[result_index[0][0]],
    }

    def convert_to_serializable(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return obj

    # Convert float32 values to float
    resultData = {key: convert_to_serializable(value) for key, value in resultData.items()}

    # Now jsonify the result
    return jsonify(resultData)

if __name__ == '__main__':
    app.run(debug=True)
