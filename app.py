# app.py (cleaned & robust)
import os
import traceback
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, session, url_for,
    send_file, flash
)
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(APP_ROOT, "logs")
MODEL_DIR = os.path.join(APP_ROOT, "model")
STATIC_GRAPH_DIR = os.path.join(APP_ROOT, "static", "graphs")
UPLOADS_DIR = os.path.join(APP_ROOT, "data")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(STATIC_GRAPH_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super-secret-key")

# ---------------- MODEL LOAD ----------------
model = None
model_path = os.path.join(MODEL_DIR, "model.pkl")
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded model:", model_path)
    except Exception as e:
        print("Failed to load model:", e)
        traceback.print_exc()
else:
    print("Model not found at", model_path, "- predictions will be disabled until model.pkl is added")

# ---------------- HELPERS ----------------
def read_csv_with_fallback(path):
    """
    Try reading CSV using a list of encodings to avoid utf-8 decode errors.
    Raises the last exception if none work.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_exc = e
    raise last_exc

def append_prediction_log(row: dict):
    """
    Append a prediction row to logs/predictions.csv. Try to also write an xlsx.
    """
    try:
        csv_path = os.path.join(LOGS_DIR, "predictions.csv")
        if not os.path.exists(csv_path):
            pd.DataFrame([row]).to_csv(csv_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)
    except Exception as e:
        print("Failed to write CSV log:", e)
        traceback.print_exc()

    # optional: create xlsx (requires openpyxl)
    try:
        csv_path = os.path.join(LOGS_DIR, "predictions.csv")
        xlsx_path = os.path.join(LOGS_DIR, "predictions.xlsx")
        if os.path.exists(csv_path):
            df_all = pd.read_csv(csv_path)
            df_all.to_excel(xlsx_path, index=False)
    except Exception as e:
        print("Failed to write XLSX log (openpyxl may be missing):", e)

def allowed_upload(filename: str):
    if not filename:
        return False
    fn = filename.lower()
    return fn.endswith(".csv") or fn.endswith(".xls") or fn.endswith(".xlsx")

def generate_graphs_from_df(df: pd.DataFrame):
    """
    Creates bar charts and returns a list of dicts: {'file': 'graphs/name.png', 'summary': '...'}
    """
    images = []
    try:
        if "dropout" not in df.columns:
            return [], "Dataset missing 'dropout' column (expected 0/1)"
        # ensure dropout numeric 0/1
        df["dropout"] = pd.to_numeric(df["dropout"], errors="coerce").fillna(0).astype(int)

        def make_bar(col, fname, title):
            path = os.path.join(STATIC_GRAPH_DIR, fname)
            plt.figure(figsize=(6,4))
            grouped = df.groupby(col)["dropout"].mean().sort_values(ascending=False)
            if grouped.empty:
                plt.text(0.5, 0.5, "No data", ha="center")
            else:
                ax = grouped.plot(kind="bar")
                ax.set_ylabel("Dropout rate (proportion)")
                ax.set_xlabel(col.capitalize())
                ax.set_ylim(0, 1)
                plt.title(title)
                for p in ax.patches:
                    h = p.get_height()
                    ax.annotate(f"{h:.2%}", (p.get_x() + p.get_width()/2, h), ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

            if not grouped.empty:
                top = grouped.idxmax()
                top_val = grouped.max()
                summary = f"Highest dropout: {top} — {top_val:.2%}"
            else:
                summary = "No data available"
            return {"file": f"graphs/{fname}", "summary": summary}

        candidates = [
            ("school", "school.png", "School-wise Dropout Rate"),
            ("area", "area.png", "Area-wise Dropout Rate"),
            ("gender", "gender.png", "Gender-wise Dropout Rate"),
            ("caste", "caste.png", "Caste-wise Dropout Rate"),
            ("standard", "standard.png", "Standard-wise Dropout Rate"),
        ]
        for col, fname, title in candidates:
            if col in df.columns:
                images.append(make_bar(col, fname, title))

        return images, None
    except Exception as e:
        print("Error generating graphs:", e)
        traceback.print_exc()
        return [], str(e)

# ---------------- ROUTES ----------------
@app.route("/")
def login():
    # if already logged in redirect to dashboard to avoid showing login again
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    user = request.form.get("username", "").strip()
    pw = request.form.get("password", "").strip()
    if user == "admin" and pw == "admin":
        session["user"] = "admin"
        return redirect(url_for("dashboard"))
    return render_template("login.html", error="Invalid credentials")

@app.route("/logout_pages")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# ---------- Prediction routes ----------
@app.route("/predict_pages")
def predict_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return render_template("result.html",
                                   prediction="Model not available",
                                   probability="N/A",
                                   school="", area="", gender="", caste="", standard="",
                                   error="Model not loaded"), 500

        school = request.form.get("school", "").strip()
        area = request.form.get("area", "").strip()
        gender = request.form.get("gender", "").strip()
        caste = request.form.get("caste", "").strip()
        standard = request.form.get("standard", "").strip()

        input_df = pd.DataFrame([{
            "school": school,
            "area": area,
            "gender": gender,
            "caste": caste,
            "standard": standard
        }])

        pred_raw = model.predict(input_df)[0]

        probability = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                if proba.shape[1] >= 2:
                    probability = float(proba[0, 1])
                else:
                    probability = float(proba[0, 0])
        except Exception:
            probability = None

        prediction_label = "Likely to Dropout" if int(pred_raw) == 1 else "Not Likely to Dropout"

        timestamp = datetime.utcnow().isoformat()
        log_row = {
            "timestamp": timestamp,
            "school": school,
            "area": area,
            "gender": gender,
            "caste": caste,
            "standard": standard,
            "prediction_raw": int(pred_raw),
            "prediction_label": prediction_label,
            "probability": probability
        }
        append_prediction_log(log_row)

        context = {
            "prediction": prediction_label,
            "probability": f"{probability:.3f}" if probability is not None else "N/A",
            "school": school,
            "area": area,
            "gender": gender,
            "caste": caste,
            "standard": standard
        }
        return render_template("result.html", **context)
    except Exception as e:
        print("Exception on /predict:", e)
        traceback.print_exc()
        return render_template("result.html",
                               prediction="Error",
                               probability="N/A",
                               school="", area="", gender="", caste="", standard="",
                               error=str(e)), 500

# ---------- Upload dataset ----------
@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if "user" not in session:
        return redirect(url_for("login"))

    message = None
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "No file uploaded"
            return render_template("upload.html", error=error)
        if not allowed_upload(file.filename):
            error = "Only CSV or Excel files are allowed"
            return render_template("upload.html", error=error)

        filename = "uploaded_dataset" + os.path.splitext(file.filename)[1]
        save_path = os.path.join(UPLOADS_DIR, filename)
        try:
            file.save(save_path)
            try:
                if filename.lower().endswith(".csv"):
                    df = read_csv_with_fallback(save_path)
                else:
                    df = pd.read_excel(save_path)
            except Exception as re:
                error = f"Failed to read uploaded file: {re}"
                print(error)
                traceback.print_exc()
                return render_template("upload.html", error=error)

            images, gerr = generate_graphs_from_df(df)
            if gerr:
                error = gerr
                return render_template("upload.html", error=error)
            message = f"Uploaded and generated {len(images)} graphs"
            return redirect(url_for("graphs_pages"))
        except Exception as e:
            error = f"Failed to save file: {e}"
            print(error)
            traceback.print_exc()
            return render_template("upload.html", error=error)

    return render_template("upload.html", message=message)

# ---------- Graphs ----------
@app.route("/graphs_pages")
def graphs_pages():
    if "user" not in session:
        return redirect(url_for("login"))

    df = None
    uploaded_csv = None
    for candidate in ["uploaded_dataset.csv", "uploaded_dataset.xls", "uploaded_dataset.xlsx", "student_data.csv"]:
        p = os.path.join(UPLOADS_DIR, candidate)
        if os.path.exists(p):
            uploaded_csv = p
            break

    if uploaded_csv:
        try:
            if uploaded_csv.lower().endswith(".csv"):
                df = read_csv_with_fallback(uploaded_csv)
            else:
                df = pd.read_excel(uploaded_csv)
        except Exception as e:
            print("Failed to read dataset:", e)
            traceback.print_exc()
            return render_template("graph_dashboard.html", images=[], error=f"Failed to read dataset: {e}")
    else:
        fallback = os.path.join(APP_ROOT, "student_data.csv")
        if os.path.exists(fallback):
            try:
                df = read_csv_with_fallback(fallback)
            except Exception as e:
                print("Failed to read fallback dataset:", e)
                traceback.print_exc()
                return render_template("graph_dashboard.html", images=[], error=f"Failed to read dataset: {e}")

    if df is None:
        return render_template("graph_dashboard.html", images=[], error="No dataset found. Upload or add student_data.csv in project root.")

    images, gerr = generate_graphs_from_df(df)
    if gerr:
        return render_template("graph_dashboard.html", images=images, error=gerr)
    return render_template("graph_dashboard.html", images=images)

# ---------- History / Download ----------
@app.route("/prediction_history")
def prediction_history():
    if "user" not in session:
        return redirect(url_for("login"))
    csv_path = os.path.join(LOGS_DIR, "predictions.csv")
    logs = []
    cols = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            cols = list(df.columns)
            logs = df.to_dict(orient="records")
        except Exception as e:
            print("Failed to read logs:", e)
            traceback.print_exc()
    return render_template("prediction_history.html", logs=logs, cols=cols)

@app.route("/download_logs/<fmt>")
def download_logs(fmt):
    if "user" not in session:
        return redirect(url_for("login"))
    csv_path = os.path.join(LOGS_DIR, "predictions.csv")
    xlsx_path = os.path.join(LOGS_DIR, "predictions.xlsx")

    if fmt == "csv":
        if not os.path.exists(csv_path):
            flash("No logs yet", "error")
            return redirect(url_for("prediction_history"))
        return send_file(csv_path, as_attachment=True, download_name="predictions.csv")
    elif fmt == "xlsx":
        if not os.path.exists(xlsx_path):
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df.to_excel(xlsx_path, index=False)
                except Exception as e:
                    print("Failed to create xlsx:", e)
                    traceback.print_exc()
                    flash("Failed to create xlsx (openpyxl required)", "error")
                    return redirect(url_for("prediction_history"))
            else:
                flash("No logs yet", "error")
                return redirect(url_for("prediction_history"))
        return send_file(xlsx_path, as_attachment=True, download_name="predictions.xlsx")
    else:
        flash("Unsupported format", "error")
        return redirect(url_for("prediction_history"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
