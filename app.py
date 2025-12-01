from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# -------------------- LOAD MODEL --------------------
model = None
model_path = os.path.join("model", "model.pkl")
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Loaded model:", model_path)
    except Exception as e:
        print("Error loading model:", e)
        model = None
else:
    print("Warning: model file not found at", model_path)
    model = None

# -------------------- LOGIN PAGE --------------------
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    user = request.form.get("username", "").strip()
    pw = request.form.get("password", "").strip()

    if user == "admin" and pw == "admin":
        session["user"] = "admin"
        return redirect("/dashboard")
    else:
        return render_template("login.html", error="Invalid Credentials")

# -------------------- DASHBOARD --------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")

# -------------------- PREDICT PAGE --------------------
@app.route("/predict_pages")
def predict_page():
    if "user" not in session:
        return redirect("/")
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/")

    if model is None:
        return render_template("predict.html", error="Model not loaded. Train the model or place model/model.pkl")

    # collect inputs
    school = request.form.get("school", "").strip()
    area = request.form.get("area", "").strip()
    gender = request.form.get("gender", "").strip()
    caste = request.form.get("caste", "").strip()
    standard = request.form.get("standard", "").strip()

    # Convert to DataFrame (IMPORTANT)
    input_df = pd.DataFrame([{
        "school": school,
        "area": area,
        "gender": gender,
        "caste": caste,
        "standard": standard
    }])

    try:
        # If model is a pipeline, it will accept the raw categorical df
        prob = None
        pred_label = model.predict(input_df)[0]
        # try to get probability if available
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
    except Exception as e:
        # return a helpful error so you can debug
        return render_template("predict.html", error=f"Prediction error: {e}")

    prediction_text = "Likely to Dropout" if int(pred_label) == 1 else "Not Likely to Dropout"
    prob_text = f"{round(prob*100,1)}%" if prob is not None else "N/A"

    # Send values to result.html
    return render_template(
        "result.html",
        prediction=prediction_text,
        probability=prob_text,
        school=school,
        area=area,
        gender=gender,
        caste=caste,
        standard=standard
    )

# -------------------- GRAPH DASHBOARD --------------------
@app.route("/graphs_pages")
def graphs_pages():
    if "user" not in session:
        return redirect("/")

    data_path = os.path.join("data", "student_data.csv")
    if not os.path.exists(data_path):
        # fall back to root CSV if user put it there
        if os.path.exists("student_data.csv"):
            data_path = "student_data.csv"
        else:
            return render_template("graph_dashboard.html", error=f"Dataset not found at {data_path}")

    # Read CSV
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return render_template("graph_dashboard.html", error=f"Could not read data file: {e}")

    # Ensure dropout column numeric 0/1
    if "dropout" in df.columns:
        # handle Yes/No or text values
        if df["dropout"].dtype == object:
            df["dropout"] = df["dropout"].map({"Yes": 1, "No": 0}).fillna(df["dropout"])
        df["dropout"] = pd.to_numeric(df["dropout"], errors="coerce").fillna(0)
    else:
        # If missing, create dummy 0 column (so plots will show 0s)
        df["dropout"] = 0

    graph_dir = os.path.join("static", "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    def safe_plot_cat(column, filename, title, rotate_xticks=False, figsize=(8,4.2)):
        """
        Create bar chart of mean(dropout) grouped by `column`, save to static/graphs/filename.
        Adds percent labels above bars and summary line with top-risk category inside title.
        Returns a tuple (saved_bool, summary_text) so the template can display text if required.
        """
        if column not in df.columns:
            return False, None

        agg = df.groupby(column)["dropout"].mean().sort_values(ascending=False)
        if agg.empty:
            return False, None

        # percent values for display
        rates = (agg * 100).round(1)
        labels = [str(x) for x in agg.index]

        plt.figure(figsize=figsize)
        ax = plt.gca()

        bars = ax.bar(labels, rates)

        # annotate each bar with percent
        for bar, pct in zip(bars, rates):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + max(1, rates.max()*0.03), f"{pct:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='600')

        # summary: top label and value
        top_idx = rates.idxmax() if hasattr(rates, "idxmax") else labels[0]
        top_val = rates.max() if hasattr(rates, "max") else float(rates.iloc[0])
        summary = f"Highest risk — {str(top_idx)} ({top_val:.0f}%)"

        ax.set_title(f"{title}\n{summary}", fontsize=12, pad=14)
        ax.set_ylabel("Dropout rate (%)")
        if rotate_xticks:
            plt.xticks(rotation=35, ha="right")

        plt.ylim(0, max(rates.max() * 1.15, 10))
        plt.tight_layout()

        save_path = os.path.join(graph_dir, filename)
        plt.savefig(save_path)
        plt.close()
        return True, summary

    # Generate required plots
    plots_info = {}

    # school
    s1, sum1 = safe_plot_cat("school", "school.png", "School-wise Dropout Rate", rotate_xticks=True)
    plots_info["school"] = {"file": "graphs/school.png", "ok": s1, "summary": sum1}

    # area
    s2, sum2 = safe_plot_cat("area", "area.png", "Area-wise Dropout Rate", rotate_xticks=False)
    plots_info["area"] = {"file": "graphs/area.png", "ok": s2, "summary": sum2}

    # gender
    s3, sum3 = safe_plot_cat("gender", "gender.png", "Gender-wise Dropout Rate", rotate_xticks=False)
    plots_info["gender"] = {"file": "graphs/gender.png", "ok": s3, "summary": sum3}

    # caste
    s4, sum4 = safe_plot_cat("caste", "caste.png", "Caste-wise Dropout Rate", rotate_xticks=True)
    plots_info["caste"] = {"file": "graphs/caste.png", "ok": s4, "summary": sum4}

    # standard / age variants
    s5 = False
    sum5 = None
    for col_name in ["standard", "age_group", "class", "age"]:
        if col_name in df.columns:
            ok, summ = safe_plot_cat(col_name, "standard.png", f"{col_name.replace('_', ' ').title()}-wise Dropout Rate", rotate_xticks=True)
            s5 = ok
            sum5 = summ
            break
    plots_info["standard"] = {"file": "graphs/standard.png", "ok": s5, "summary": sum5}

    # Build images list (only include those actually created)
    images = []
    for key in ["school", "area", "gender", "caste", "standard"]:
        info = plots_info.get(key)
        if info and info["ok"]:
            images.append({"file": info["file"], "summary": info["summary"]})

    return render_template("graph_dashboard.html", images=images)

# -------------------- LOGOUT --------------------
@app.route("/logout_pages")
def logout():
    session.clear()
    return redirect("/")

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)
