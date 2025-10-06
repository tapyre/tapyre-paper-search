from flask import Flask
from src.routes.chunks_bp import chunk_blueprint

app = Flask(__name__)

app.register_blueprint(chunk_blueprint, url_prefix="/chunks")

if __name__ == "__main__":
    print("Running api.py...")
    app.run(host="0.0.0.0", port=8000, debug=True)