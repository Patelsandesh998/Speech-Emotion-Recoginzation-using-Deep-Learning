import os
import os.path as osp
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
try:
    from pydub import AudioSegment  # requires ffmpeg installed on system
except Exception:
    AudioSegment = None

# Local imports
from predictions import make_predictions


ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac", "webm"}


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    def is_allowed(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory("uploads", filename)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not is_allowed(file.filename):
            return jsonify({"error": "Unsupported file type"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join("uploads", filename)
        file.save(save_path)

        # If not a WAV file, try converting to WAV so librosa can load reliably
        process_path = save_path
        root, ext = osp.splitext(filename)
        if ext.lower() != ".wav":
            if AudioSegment is None:
                return jsonify({
                    "ok": False,
                    "error": "Non-WAV upload received but audio converter (pydub/ffmpeg) is not available. Install with: pip install pydub and brew install ffmpeg"
                }), 400
            try:
                audio = AudioSegment.from_file(save_path)
                wav_path = os.path.join("uploads", f"{root}.wav")
                # standardize sample rate/channels for model
                audio = audio.set_frame_rate(44100).set_channels(1)
                audio.export(wav_path, format="wav")
                process_path = wav_path
            except Exception as conv_exc:
                return jsonify({
                    "ok": False,
                    "error": f"Failed to convert audio to WAV: {str(conv_exc)}"
                }), 400

        try:
            preds = make_predictions(process_path)
            return jsonify({
                "ok": True,
                "predictions": preds,
                "filename": osp.basename(process_path)
            })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    # Host on localhost:5000 by default
    app.run(host="127.0.0.1", port=5000, debug=True)


