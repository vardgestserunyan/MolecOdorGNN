from flask import Flask, render_template, request, jsonify
from datetime import datetime
from utils import gnn_runner
import os 


app = Flask(__name__)



@app.route('/predictor', methods=["GET", "POST"])
def predictor():
    data, show_results = {}, False
    if request.method == "POST":
        smiles_input = request.form.get('smiles_input')
        smiles_input_list = smiles_input.splitlines()
        prediction = gnn_runner(smiles_input_list)
        date, time = datetime.now().date(), datetime.now().time()
        data = {'user_input': smiles_input,
                'smiles_input_list':smiles_input_list,
                'prediction':prediction,
                'date':date,
                'time':time
                }
        show_results = True
    return render_template('/predictor.html', data=data, show_results=show_results)

@app.route('/api/predictor', methods=["POST"])
def api_predictor():
    smiles_input_list = request.get_json(force=True).get('smiles_input_list')
    prediction = gnn_runner(smiles_input_list)
    timestamp = datetime.now()
    data = {'smiles_input_list':smiles_input_list,
            'prediction':prediction,
            'timestamp':str(timestamp),
            }
    return jsonify(data)

if __name__ == "__main__":
    host, port, debug = "0.0.0.0", "8080", True
    port = int(os.environ.get("PORT", 8080))
    app.run(host=host, port=port, debug=debug)