from flask import Flask, render_template, request, make_response, jsonify
import pickle
import io
import csv
from io import StringIO
import pandas as pd
from sklearn.metrics import silhouette_score

global s,s1,s2
app = Flask(__name__)
model = pickle.load(open('kmeans.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/defaults',methods=['POST'])

def defaults():

    return render_template('index.html')


def transform(text_file_contents):

    return text_file_contents.replace("=", ",")



@app.route('/transform', methods=["POST"])

def transform_view():

    f = request.files['data_file']

    if not f:

        return "No file"



    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)

    csv_input = csv.reader(stream)

    print(csv_input)

    for row in csv_input:

        print(row)



    stream.seek(0)

    result = transform(stream.read())

    
    if request.form['cls'] == 'kmeans':
        df = pd.read_csv(StringIO(result))
    
        loaded_model = pickle.load(open('kmeans.pkl', 'rb'))
        kmeans = loaded_model.fit(df)
        df['Cluster'] = kmeans.predict(df)
        response = make_response(df.to_csv())
        s = silhouette_score(df, kmeans.labels_)
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"

        
    elif request.form['cls'] == 'agglomerative':
        df = pd.read_csv(StringIO(result))
        
        loaded_model = pickle.load(open('agglomerative.pkl', 'rb'))

        df['Cluster'] = loaded_model.fit_predict(df)
        response = make_response(df.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    
    elif request.form['cls'] == 'dbscan':
        df = pd.read_csv(StringIO(result))
    
        loaded_model = pickle.load(open('dbscan.pkl', 'rb'))

        df['Cluster'] = loaded_model.fit_predict(df)
        response = make_response(df.to_csv())

        response.headers["Content-Disposition"] = "attachment; filename=result.csv"



    return response



if __name__=="__main__":
    app.run(debug=True)