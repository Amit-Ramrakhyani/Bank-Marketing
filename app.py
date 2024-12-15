from flask import Flask, render_template, request
from utils import load_data, create_dataframe, preprocess_data, predict_y

app = Flask(__name__)

model_path = 'data/balanced/best_classifier.pkl'
input_cols_path = 'data/balanced/input_cols.sav'
mean_path = 'data/balanced/df_mean.csv'
scaler_path = 'data/balanced/scaler.sav'

model, input_cols, mean, scaler = load_data(model_path, input_cols_path, mean_path, scaler_path)

# Data dictionary format
# data = {'age': [30],
#         'job': ['admin.'],
#         'marital': ['single'],
#         'education': ['university.degree'],
#         'default': ['no'],
#         'housing': ['no'],
#         'loan': ['no'],
#         'contact': ['cellular'],
#         'month': ['may'],
#         'day_of_week': ['mon'],
#         'duration': [487],
#         'campaign': [2],
#         'pdays': [999],
#         'previous': [0],
#         'poutcome': ['nonexistent'],
#         'emp.var.rate': [-1.8],
#         'cons.price.idx': [92.893],
#         'cons.conf.idx': [-46.2],
#         'euribor3m': [1.299],
#         'nr.employed': [5099.1],
#         'y': ['no']}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        
        form_data = {
            key: [float(value)] if value.replace('.', '', 1).isdigit() else [value]
            for key, value in form_data.items()
        }
        
        df = create_dataframe(form_data, input_cols)
        X = preprocess_data(df, input_cols, mean, scaler)
        
        pred = predict_y(X, model).astype(int)
        
        return render_template(
            'index.html',
            prediction="Subscribed" if pred == 1 else "Not Subscribed"
        )

if __name__ == '__main__':
    app.run(debug=True)
