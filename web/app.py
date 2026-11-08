"""Website klasifikasi penyakit daun padi (ResNet-50 + Grad-CAM).

Preprocessing dan Grad-CAM di sini dibuat identik dengan notebook
Rice_Leaf_Disease-2.ipynb agar hasil prediksi sama dengan saat penelitian.

Menjalankan:
    python app.py            # lalu buka http://127.0.0.1:5000
"""

import base64
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.resnet50 import preprocess_input  # noqa: E402

from disease_info import DISEASE_INFO, get_info  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "model" / "resnet50_daun_padi.keras"))
CLASS_NAMES_PATH = Path(os.environ.get("CLASS_NAMES_PATH", BASE_DIR / "model" / "class_names.json"))

IMG_SIZE = 224                      # sama dengan IMG_SIZE di notebook
LAST_CONV = "conv5_block3_out"      # layer Grad-CAM di notebook
HEAD_LAYERS = ["gap", "dropout_1", "dense_relu", "dropout_2", "output_softmax"]
LOW_CONFIDENCE = 0.70               # di bawah ini tampilkan peringatan
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
DISPLAY_SIZE = 448                  # ukuran gambar hasil untuk ditampilkan

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Classifier:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.conv_model = None
        self.head_layers = []
        self.error = None
        self._lock = threading.Lock()

    def load(self):
        if not MODEL_PATH.exists():
            self.error = (
                f"File model tidak ditemukan: {MODEL_PATH}. Salin resnet50_daun_padi.keras "
                "dari Google Drive (PROJECT/hasil_skripsi/) ke folder web/model/."
            )
            print("[PERINGATAN]", self.error)
            return

        print(f"Memuat model dari {MODEL_PATH} ...")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        if CLASS_NAMES_PATH.exists():
            self.class_names = json.loads(CLASS_NAMES_PATH.read_text())
        else:
            # Urutan alfabetis, sama seperti sorted() di notebook.
            self.class_names = sorted(DISEASE_INFO.keys())
            print(f"[PERINGATAN] {CLASS_NAMES_PATH} tidak ada, memakai {self.class_names}")

        n_out = self.model.output_shape[-1]
        if n_out != len(self.class_names):
            raise ValueError(f"Model punya {n_out} output, tetapi class_names berisi {len(self.class_names)} kelas.")

        base = self._find_backbone()
        self.conv_model = tf.keras.Model(base.inputs, base.get_layer(LAST_CONV).output)
        self.head_layers = [self.model.get_layer(n) for n in HEAD_LAYERS]

        # Pemanasan agar permintaan pertama tidak lambat.
        self.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
        print("Model siap. Kelas:", self.class_names)

    def _find_backbone(self):
        try:
            return self.model.get_layer("resnet50")
        except ValueError:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.Model) and any(l.name == LAST_CONV for l in layer.layers):
                    return layer
        raise ValueError(f"Backbone dengan layer {LAST_CONV} tidak ditemukan di model.")

    @property
    def ready(self):
        return self.model is not None

    def predict(self, img_float):
        """img_float: array (224, 224, 3) nilai 0-255 hasil resize (belum preprocess_input)."""
        arr = preprocess_input(img_float.copy())[None, ...]
        with self._lock:
            with tf.GradientTape() as tape:
                conv_out = self.conv_model(arr, training=False)
                tape.watch(conv_out)
                x = conv_out
                for layer in self.head_layers:
                    x = layer(x, training=False)
                preds = x
                class_index = int(tf.argmax(preds[0]))
                score = preds[:, class_index]
            grads = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0].numpy()
        return preds.numpy()[0], class_index, cam


classifier = Classifier()


# --------------------------------------------------------------------------- #
# Utilitas gambar
# --------------------------------------------------------------------------- #
def decode_and_resize(raw: bytes) -> np.ndarray:
    """Sama dengan load_img() di notebook: decode 3 kanal lalu resize 224x224."""
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.numpy()


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """Pendekatan colormap 'jet' matplotlib, x di [0, 1] → RGB [0, 1]."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def to_data_url(arr_uint8: np.ndarray) -> str:
    im = Image.fromarray(arr_uint8).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay_gradcam(img_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = jet_colormap(cam) * 255.0
    out = img_uint8.astype(np.float32) * (1 - alpha) + heat * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Rute
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    order = classifier.class_names or sorted(DISEASE_INFO.keys())
    diseases = [{"key": k, **get_info(k)} for k in order]
    return render_template("index.html", diseases=diseases, threshold=int(LOW_CONFIDENCE * 100))


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=classifier.ready,
        classes=classifier.class_names,
        error=classifier.error,
    )


@app.post("/api/predict")
def predict():
    if not classifier.ready:
        return jsonify(error=classifier.error or "Model belum dimuat."), 503

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Tidak ada file gambar yang diunggah."), 400
    if Path(file.filename).suffix.lower() not in ALLOWED_EXT:
        return jsonify(error="Format tidak didukung. Gunakan JPG, JPEG, atau PNG."), 400

    raw = file.read()
    t0 = time.perf_counter()
    try:
        img = decode_and_resize(raw)
    except (tf.errors.InvalidArgumentError, ValueError):
        return jsonify(error="File tidak dapat dibaca sebagai gambar."), 400

    probs, idx, cam = classifier.predict(img)
    img_uint8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
    names = classifier.class_names
    label = names[idx]
    confidence = float(probs[idx])

    ranking = sorted(
        ({"class": c, "name": get_info(c)["name"], "probability": float(p)} for c, p in zip(names, probs)),
        key=lambda r: -r["probability"],
    )
    return jsonify(
        label=label,
        label_id=get_info(label)["name"],
        confidence=confidence,
        low_confidence=confidence < LOW_CONFIDENCE,
        probabilities=ranking,
        info=get_info(label),
        input_image=to_data_url(img_uint8),
        gradcam=to_data_url(overlay_gradcam(img_uint8, cam)),
        elapsed_ms=round((time.perf_counter() - t0) * 1000),
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Ukuran file melebihi 10 MB."), 413


classifier.load()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)
