from flask import Flask, jsonify, request
from flask_cors import CORS
from model import downloadData, predict, FeatureEngineering

app = Flask(__name__)
CORS(app)

@app.route('/')
def display_status():
    return "✅ Backend server is online"

@app.route('/predict', methods=['POST'])
def call_model():
    fetched_data_from_front_end = request.get_json()
    ticker = fetched_data_from_front_end.get('stock')

    if not ticker:
        return jsonify({'error': 'Missing stock symbol'}), 400
    
    stock_data = downloadData(ticker)
    if stock_data is None:
        return jsonify({'error' : 'Invalid stock symbol or no data found'}), 404
    
    engineered_data = FeatureEngineering(stock_data)
    result = predict(engineered_data)

    return jsonify(result)

if __name__ == '__main__':
    print('Backend server is online!')
    app.run(debug=True)

