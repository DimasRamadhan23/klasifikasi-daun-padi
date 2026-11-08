"""Konversi model Keras (.keras) dari notebook menjadi model TensorFlow.js untuk website.

Model hasil konversi menerima citra mentah 224x224 (nilai 0-255) dan mengeluarkan:
  - probabilities : [1, K]          probabilitas softmax tiap kelas
  - gradcam       : [1, 224, 224, 1] heatmap Grad-CAM (0-1) untuk kelas teratas

preprocess_input ResNet-50 dan Grad-CAM sudah tertanam di dalam graf, sehingga
JavaScript cukup mengirim piksel hasil resize. Grad-CAM dihitung secara analitik
(rantai turunan kepala GAP -> Dense ReLU -> Dense Softmax), hasilnya identik dengan
versi GradientTape di notebook. Skrip ini memeriksa kesamaan itu sebelum menyimpan.

Pemakaian:
    python tools/konversi_model.py \
        --model  path/ke/resnet50_daun_padi.keras \
        --classes path/ke/class_names.json \
        --out    website/model
    # tambahkan --float32 untuk bobot penuh (~94 MB) alih-alih float16 (~47 MB)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.resnet50 import preprocess_input  # noqa: E402

IMG_SIZE = 224
LAST_CONV = "conv5_block3_out"


def find_backbone(model):
    try:
        return model.get_layer("resnet50")
    except ValueError:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and any(l.name == LAST_CONV for l in layer.layers):
                return layer
    raise ValueError(f"Backbone dengan layer {LAST_CONV} tidak ditemukan.")


class Exporter(tf.Module):
    def __init__(self, model):
        super().__init__()
        base = find_backbone(model)
        self.conv_model = tf.keras.Model(base.inputs, base.get_layer(LAST_CONV).output)
        d1, d2 = model.get_layer("dense_relu"), model.get_layer("output_softmax")
        self.w1 = tf.constant(d1.kernel.numpy())  # [2048, 256]
        self.b1 = tf.constant(d1.bias.numpy())
        self.w2 = tf.constant(d2.kernel.numpy())  # [256, K]
        self.b2 = tf.constant(d2.bias.numpy())

    @tf.function(input_signature=[tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="citra")])
    def serve(self, x):
        f = self.conv_model(preprocess_input(x), training=False)        # [1, 7, 7, 2048]
        g = tf.reduce_mean(f, axis=[1, 2])                              # GAP
        h_pre = tf.matmul(g, self.w1) + self.b1                         # Dense 256 (Dropout nonaktif)
        h = tf.nn.relu(h_pre)
        p = tf.nn.softmax(tf.matmul(h, self.w2) + self.b2)              # [1, K]

        # d p_c / d f  untuk c = kelas teratas
        c = tf.argmax(p[0])
        k = tf.shape(p)[1]
        p_c = tf.gather(p[0], c)
        dz = p_c * (tf.one_hot(c, k) - p[0])                            # turunan softmax
        dh = tf.linalg.matvec(self.w2, dz) * tf.cast(h_pre[0] > 0, tf.float32)
        dg = tf.linalg.matvec(self.w1, dh)                              # [2048]
        hw = tf.cast(tf.shape(f)[1] * tf.shape(f)[2], tf.float32)
        alpha = dg / hw                                                 # rata-rata gradien per kanal

        cam = tf.nn.relu(tf.reduce_sum(f[0] * alpha, axis=-1))
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam = tf.image.resize(cam[None, :, :, None], (IMG_SIZE, IMG_SIZE))
        return {"probabilities": p, "gradcam": cam}


def gradcam_tape(model, x):
    """Versi notebook (GradientTape) untuk pembanding."""
    base = find_backbone(model)
    conv_model = tf.keras.Model(base.inputs, base.get_layer(LAST_CONV).output)
    head = [model.get_layer(n) for n in ["gap", "dropout_1", "dense_relu", "dropout_2", "output_softmax"]]
    arr = preprocess_input(x.copy())
    with tf.GradientTape() as tape:
        conv_out = conv_model(arr, training=False)
        tape.watch(conv_out)
        y = conv_out
        for layer in head:
            y = layer(y, training=False)
        idx = int(tf.argmax(y[0]))
        score = y[:, idx]
    grads = tape.gradient(score, conv_out)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * w[:, None, None, :], axis=-1)[0])
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0].numpy()
    return model(arr, training=False).numpy()[0], cam


def verify(model, exporter):
    rng = np.random.default_rng(0)
    for i in range(3):
        x = rng.uniform(0, 255, (1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
        out = exporter.serve(tf.constant(x))
        p_ref, cam_ref = gradcam_tape(model, x)
        dp = np.abs(out["probabilities"].numpy()[0] - p_ref).max()
        dc = np.abs(out["gradcam"].numpy()[0, ..., 0] - cam_ref).max()
        print(f"  cek {i + 1}: selisih probabilitas = {dp:.2e}, selisih Grad-CAM = {dc:.2e}")
        if dp > 1e-4 or dc > 1e-3:
            raise SystemExit("GAGAL: hasil ekspor berbeda dengan model asli.")


def convert(saved_dir, out_dir, float16):
    # tensorflowjs meng-import paket-paket ini di awal, tetapi tidak memakainya untuk konversi
    # SavedModel. Paket tersebut tidak kompatibel dengan TF 2.20 (atau berat), jadi diganti
    # modul kosong.
    for name in ("tensorflow_decision_forests", "tensorflow_hub", "jax", "jax.experimental",
                 "jax.experimental.jax2tf"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["jax.experimental"].jax2tf = sys.modules["jax.experimental.jax2tf"]
    from tensorflowjs.converters import tf_saved_model_conversion_v2 as conv

    conv.convert_tf_saved_model(
        str(saved_dir),
        str(out_dir),
        signature_def="serving_default",
        saved_model_tags="serve",
        quantization_dtype_map={"float16": "*"} if float16 else None,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path resnet50_daun_padi.keras")
    ap.add_argument("--classes", required=True, help="path class_names.json")
    ap.add_argument("--out", default="website/model", help="folder keluaran (default: website/model)")
    ap.add_argument("--float32", action="store_true", help="simpan bobot float32 penuh (lebih besar)")
    args = ap.parse_args()

    classes = json.loads(Path(args.classes).read_text())
    print("Memuat model:", args.model)
    model = tf.keras.models.load_model(args.model, compile=False)
    if model.output_shape[-1] != len(classes):
        raise SystemExit(f"Model punya {model.output_shape[-1]} output, class_names berisi {len(classes)}.")

    exporter = Exporter(model)
    print("Memeriksa kesamaan dengan Grad-CAM notebook ...")
    verify(model, exporter)

    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "saved_model"
        tf.saved_model.save(exporter, str(saved), signatures={"serving_default": exporter.serve})
        if out.exists():
            for f in out.glob("*"):
                if f.name == "model.json" or f.name.startswith("group") or f.name == "metadata.json":
                    f.unlink()
        out.mkdir(parents=True, exist_ok=True)
        print("Mengonversi ke TensorFlow.js ...")
        convert(saved, out, float16=not args.float32)

    meta = {
        "classes": classes,
        "img_size": IMG_SIZE,
        "last_conv": LAST_CONV,
        "weights": "float32" if args.float32 else "float16",
        "inputs": "citra [1,224,224,3] float32 0-255 (preprocess_input di dalam graf)",
        "outputs": ["probabilities", "gradcam"],
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    size = sum(f.stat().st_size for f in out.glob("*")) / 1e6
    print(f"Selesai. {len(list(out.glob('*')))} file, {size:.1f} MB di {out}")


if __name__ == "__main__":
    main()
