from flask import Flask, request, jsonify
from data_encoder import encode_data
from telephone.sender import process_and_play
import os

app = Flask(__name__)

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../schemas/sensor_data.avsc")

@app.route("/audio", methods=["POST"])
def send_audio():
    try:
        data = request.get_json()
        avro_bytes = encode_data(data, SCHEMA_PATH)

        process_and_play(avro_bytes)

        return jsonify({"status": "success", "message": "Audio played successfully"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


