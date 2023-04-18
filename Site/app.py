from flask import Flask, render_template, request, url_for, redirect, Response
import sqlite3 as sql
import joblib
import pandas as pd
import sqlite3
import csv

### DATABASE CREATION ##########################
################################################
conn = sqlite3.connect('credit_card_fraud.db')
cur = conn.cursor()

cur.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        distance_from_home REAL,
        distance_from_last_transaction REAL,
        ratio_to_median_purchase_price REAL,
        repeat_retailer INTEGER,
        used_chip INTEGER,
        used_pin INTEGER,
        online_order INTEGER,
        fraud INTEGER
    )
''')

conn.commit()
conn.close()
################################################

values = []
transaction_data = {}
result = None
fraud_probability = None

model = joblib.load('xgbc_model.joblib')

# Create a new Flask app instance
app = Flask(__name__)

# Define the host for the app
host = 'http://127.0.0.1:5000/'

# Define the index route
@app.route('/')
def index():
    # Render the index.html template
    return render_template('index.html')


@app.route("/analyze", methods=["POST", 'GET'])
def analyze():

    error = None
    if request.method == 'POST':
        try:
            distance_from_home = float(request.form["distance_from_home"])
            distance_from_last_transaction = float(request.form["distance_from_last_transaction"])
            ratio_to_median_purchase_price = float(request.form["ratio_to_median_purchase_price"])
            repeat_retailer = 1 if request.form.get("repeat") == "yes" else 0
            used_chip = 1 if request.form.get("chip") == "yes" else 0
            used_pin_number = 1 if request.form.get("pin") == "yes" else 0
            online_order = 1 if request.form.get("online") == "yes" else 0
        except ValueError:
            return render_template('index.html', error = "Please fill out all fields.")


        local_values=[distance_from_home, 
                distance_from_last_transaction, 
                ratio_to_median_purchase_price, 
                repeat_retailer, 
                used_chip, 
                used_pin_number, 
                online_order]
        for value in local_values:
            values.append(value)


        global result
        global fraud_probability
        result, fraud_probability = predict(local_values) 

        form_data = {
        'distance_from_home': request.form['distance_from_home'],
        'distance_from_last_transaction': request.form['distance_from_last_transaction'],
        'ratio_to_median_purchase_price': request.form['ratio_to_median_purchase_price'],
        'repeat_retailer': request.form['repeat'],
        'used_chip': request.form['chip'],
        'used_pin': request.form['pin'],
        'online_order': request.form['online']
    }
        global transaction_data
        for key in form_data.keys():
            transaction_data[key] = form_data[key]

    return render_template('analyze.html', error=error, result = result, transaction=form_data, fraud_probability=fraud_probability)


def predict(values):
    # Define the feature names in the expected order
    feature_names = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
                     'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
    
    # Create a dictionary with the input values and feature names
    input_dict = dict(zip(feature_names, values))
    
    # Create a pandas DataFrame from the input dictionary
    input_df = pd.DataFrame([input_dict])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_df)
    fraud_probability = model.predict_proba(input_df)
    # Return the prediction
    return prediction[0], round(fraud_probability[0][1]*100, 3)


@app.route('/save_transaction_as_fraud', methods=['POST'])
def save_transaction_as_fraud():
    global result
    global transaction_data
    global fraud_probability
    if values:
        # connect to database
        conn = sqlite3.connect('credit_card_fraud.db')
        cur = conn.cursor()
    
        # insert values into table
        cur.execute('''
            INSERT INTO transactions (distance_from_home, distance_from_last_transaction, 
            ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin, online_order, fraud)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (values[0], values[1], values[2], values[3], values[4], values[5], values[6], 1))
    
        conn.commit()
        conn.close()

        while values:
            values.pop()

        return render_template('analyze.html', transaction=transaction_data, message = "Successfully added to Database.", fraud_probability=fraud_probability, result=result)
    else:
        return render_template('analyze.html', transaction=transaction_data, message = "Already added to Database.", fraud_probability=fraud_probability, result=result)


@app.route('/save_transaction_as_valid', methods=['POST'])
def save_transaction_as_valid():
    global result
    global transaction_data
    global fraud_probability
    if values:
        conn = sqlite3.connect('credit_card_fraud.db')
        cur = conn.cursor()
    
        cur.execute('''
            INSERT INTO transactions (distance_from_home, distance_from_last_transaction, 
            ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin, online_order, fraud)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (values[0], values[1], values[2], values[3], values[4], values[5], values[6], 0))
    
        conn.commit()
        conn.close() 

        while values:
            values.pop()

        return render_template('analyze.html', transaction=transaction_data, message = "Successfully added to Database.", fraud_probability=fraud_probability, result= result)
    else:
        return render_template('analyze.html', transaction=transaction_data, message = "Already added to Database.", fraud_probability=fraud_probability, result=result)


@app.route('/view_database', methods=['POST'])
def view_database():
    conn = sqlite3.connect('credit_card_fraud.db')
    cur = conn.cursor()

    cur.execute('SELECT * FROM transactions')
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    transactions = [dict(zip(columns, row)) for row in rows]

    conn.close()

    return render_template('view.html', transactions=transactions)


@app.route('/clear_database', methods=['POST'])
def clear_database():
    conn = sqlite3.connect('credit_card_fraud.db')
    cur = conn.cursor()

    cur.execute('DELETE FROM transactions')
    conn.commit()

    cur.execute('SELECT * FROM transactions')
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    transactions = [dict(zip(columns, row)) for row in rows]

    conn.close()

    return render_template('view.html', transactions=transactions)


@app.route('/download_database', methods=['POST'])
def download_database():
    conn = sqlite3.connect('credit_card_fraud.db')
    cur = conn.cursor()

    cur.execute('SELECT * FROM transactions')
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    transactions = [dict(zip(columns, row)) for row in rows]

    conn.close()

    # Create a CSV file and write transactions data into it
    with open('transactions.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(transactions)

    # Prepare the response for downloading the CSV file
    def generate():
        with open('transactions.csv', 'rb') as csvfile:
            while True:
                data = csvfile.read(1024)
                if not data:
                    break
                yield data

    return Response(generate(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=transactions.csv'})

# Run the program
if __name__ == "__main__":
    app.run(debug=True)


