from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model_path = 'pricedetectmodel.h5'
model = load_model(model_path)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    img_file = request.files['file']
    if img_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img = Image.open(io.BytesIO(img_file.read()))  # Load image using PIL.Image

        img = img.resize((224, 224))  # Resize image to (224, 224)
        img = img.convert('RGB')  # Convert image to RGB format
        img = np.array(img)  # Convert PIL image to NumPy array

        # Normalize pixel values
        img = img.astype(np.float32) / 255.0

        img_encoded = base64.b64encode(img).decode('utf-8')
        img_src = f"data:image/jpeg;base64,{img_encoded}"


        img = np.expand_dims(img, axis=0)  # Add batch dimension
        prediction = model.predict(img)
        # predicted_price = str(prediction[0][0])  # Get predicted price
        predicted_price = '$ {:.2f}'.format(prediction[0][0])  # Get predicted price

        return render_template('index.html', prediction_result=predicted_price, uploaded_image=img_src)
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
