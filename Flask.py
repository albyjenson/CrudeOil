from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = Flask(__name__)

# Load the saved model architecture from JSON file
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = tf.keras.models.model_from_json(loaded_model_json)

# Load weights into the loaded model
model.load_weights('crude_oil_lstm_model.h5')

model.compile(optimizer='adam', loss='mean_squared_error')

# Define the scaler (assuming the same as used during training)
scaler = MinMaxScaler(feature_range=(0, 1))

def create_dataset(data, time_step=100):
    X = []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
    X = np.array(X)
    
    return X



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json()
        
        # Ensure data is provided
        if 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        input_data = data['data']
        
        # Ensure the input data is not empty
        if not input_data:
            return jsonify({'error': 'Empty data provided'}), 400
        
        # Convert input data to a DataFrame
        df_input = pd.DataFrame(input_data)
        
        # Ensure 'CloseCrude' column is present
        if 'CloseCrude' not in df_input.columns:
            return jsonify({'error': 'Missing CloseCrude column'}), 400
        
        # Check if the number of data points is sufficient
        if len(df_input) < 100:
            return jsonify({'error': 'Insufficient data points. Need at least 100 data points.'}), 400
        
        # Prepare data
        scaled_data = scaler.fit_transform(df_input['CloseCrude'].values.reshape(-1,1))
        
        # Define time step (number of previous days to use for prediction)
        time_step = 100
        X= create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Predict
        predicted_price = model.predict(X)  # Predicting only the last sequence
        predicted_price = scaler.inverse_transform(predicted_price)
        
        # Return the prediction
        return jsonify({'predicted_price': predicted_price.flatten().tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
