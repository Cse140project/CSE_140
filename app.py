from flask import Flask, render_template, request, redirect, session, url_for
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "supersecret"

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("model/model.pkl", "rb"))

# -------------------- LOGIN PAGE --------------------
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    user = request.form["username"]
    pw = request.form["password"]

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
    school = request.form["school"]
    area = request.form["area"]
    gender = request.form["gender"]
    caste = request.form["caste"]
    standard = request.form["standard"]

    # Convert to DataFrame (IMPORTANT)
    input_df = pd.DataFrame([{
        "school": school,
        "area": area,
        "gender": gender,
        "caste": caste,
        "standard": standard
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_text = "Likely to Dropout" if prediction == 1 else "Not Likely to Dropout"

    # Send values to result.html
    return render_template(
        "result.html",
        prediction=prediction_text,
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

    df = pd.read_csv("student_data.csv")

    graph_path = "static/graphs"
    os.makedirs(graph_path, exist_ok=True)

    # School graph
    plt.figure()
    df.groupby("school")["dropout"].mean().plot(kind="bar")
    plt.title("School-wise Dropout Rate")
    plt.ylabel("Dropout Rate")
    plt.savefig(f"{graph_path}/school.png")
    plt.close()

    # Area graph
    plt.figure()
    df.groupby("area")["dropout"].mean().plot(kind="bar")
    plt.title("Area-wise Dropout Rate")
    plt.ylabel("Dropout Rate")
    plt.savefig(f"{graph_path}/area.png")
    plt.close()

    # Gender graph
    plt.figure()
    df.groupby("gender")["dropout"].mean().plot(kind="bar")
    plt.title("Gender-wise Dropout Rate")
    plt.ylabel("Dropout Rate")
    plt.savefig(f"{graph_path}/gender.png")
    plt.close()

    return render_template("graph_dashboard.html")

# -------------------- LOGOUT --------------------
@app.route("/logout_pages")
def logout():
    session.clear()
    return redirect("/")

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)
