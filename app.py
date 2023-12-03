from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np

app = Flask(__name__)

# Load your pre-trained model
# model = tf.keras.models.load_model('model/finetune_model.pth')

# Defining a route for receiving image data and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive image file from the Android app
        if 'image_data' not in request.files:
            return jsonify({'error': 'No image file provided'})

        image_file = request.files['image_data']
        
        # Check if the file is of an allowed type (e.g., image)
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if image_file.filename.split('.')[-1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Only allowed: png, jpg, jpeg, gif'})

        # You can save the image file to a specific location if needed
        # image_file.save('path/to/save/image.jpg')

        # processed_image = preprocess_image(image_file)

        # Make predictions using your model
        prediction = make_prediction(image_file)

        # Return the result as JSON
        return jsonify({'result': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

# def preprocess_image(image_file):
#     processed_image = ...
#     return processed_image

# Make predictions using model
def make_prediction(processed_image):
    # prediction = model.predict(np.expand_dims(processed_image, axis=0))
    
    prediction = 'hehe'
    return prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
