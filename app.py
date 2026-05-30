import os
import webbrowser
from flask import Flask, jsonify, request, render_template_string
import pandas as pd
from Prediction import train_and_forecast_programmatic

app = Flask(__name__)

@app.route('/')
def home():
    try:
        # Resolve templates path relative to this script
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'index.html')
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template_string(content)
    except FileNotFoundError:
        return "Frontend template file not found. Please check templates/index.html.", 404

@app.route('/api/historical', methods=['GET'])
def get_historical():
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gold_prices_1995-2026.csv')
        df = pd.read_csv(csv_path)
        df = df.dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        data = []
        for _, row in df.iterrows():
            data.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'price': float(row['Gold_Price_USD_YFinance'])
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        req_data = request.get_json() or {}
        model_name = req_data.get('model_name', 'Linear Regression')
        forecast_months = int(req_data.get('forecast_months', 12))
        params = req_data.get('params', {})
        
        # Execute programmatic ML pipeline
        results = train_and_forecast_programmatic(model_name, forecast_months, params)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = 5000
    # Run server locally. Browser opens automatically.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
