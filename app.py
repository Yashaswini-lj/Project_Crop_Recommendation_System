import os
import sqlite3
from datetime import datetime
from functools import wraps

import joblib
import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "features.json")

os.makedirs(DB_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "agriguide-dev-secret-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# The ML model is trained ONLY on these seven numeric features.
ML_FEATURES = ["N", "P", "K", "temperature", "humidity", "rainfall", "ph"]

CROP_INFO = {
    "rice": {"season": ["Kharif"], "soil": ["Alluvial", "Clay", "Loamy"], "water": ["High", "Medium"],
             "temp": (20, 35), "humidity": (60, 95), "rainfall": (150, 300), "ph": (5.0, 7.5),
             "reason": "Rice generally fits warm, humid conditions with good water availability."},
    "maize": {"season": ["Kharif", "Rabi"], "soil": ["Alluvial", "Loamy", "Black"], "water": ["Medium", "High"],
             "temp": (18, 32), "humidity": (50, 85), "rainfall": (50, 150), "ph": (5.5, 7.5),
             "reason": "Maize is compatible with moderate warmth, balanced moisture and fertile soil."},
    "chickpea": {"season": ["Rabi"], "soil": ["Loamy", "Black", "Red"], "water": ["Low", "Medium"],
                 "temp": (18, 30), "humidity": (40, 70), "rainfall": (50, 100), "ph": (6.0, 8.0),
                 "reason": "Chickpea is commonly associated with cooler Rabi conditions and moderate moisture."},
    "kidney beans": {"season": ["Kharif"], "soil": ["Loamy", "Red"], "water": ["Medium", "High"],
                     "temp": (15, 30), "humidity": (50, 80), "rainfall": (60, 150), "ph": (5.5, 7.0),
                     "reason": "Kidney beans prefer moderate temperatures and well-drained fertile soil."},
    "pigeon peas": {"season": ["Kharif"], "soil": ["Red", "Black", "Loamy"], "water": ["Low", "Medium"],
                    "temp": (20, 35), "humidity": (45, 80), "rainfall": (60, 150), "ph": (5.5, 8.0),
                    "reason": "Pigeon peas can suit warm conditions with moderate rainfall and drainage."},
    "moth beans": {"season": ["Kharif"], "soil": ["Sandy", "Loamy", "Red"], "water": ["Low", "Medium"],
                   "temp": (25, 40), "humidity": (30, 70), "rainfall": (30, 100), "ph": (6.0, 8.5),
                   "reason": "Moth bean is comparatively tolerant of warm and drier conditions."},
    "mung bean": {"season": ["Kharif", "Zaid"], "soil": ["Loamy", "Sandy", "Red"], "water": ["Low", "Medium"],
                  "temp": (25, 35), "humidity": (40, 80), "rainfall": (50, 100), "ph": (6.0, 7.5),
                  "reason": "Mung bean fits warm conditions and moderate water availability."},
    "black gram": {"season": ["Kharif", "Rabi"], "soil": ["Loamy", "Black", "Red"], "water": ["Low", "Medium"],
                   "temp": (20, 35), "humidity": (50, 80), "rainfall": (50, 100), "ph": (6.0, 7.5),
                   "reason": "Black gram generally performs under warm conditions with moderate moisture."},
    "lentil": {"season": ["Rabi"], "soil": ["Loamy", "Alluvial"], "water": ["Low", "Medium"],
               "temp": (15, 28), "humidity": (40, 70), "rainfall": (30, 100), "ph": (6.0, 8.0),
               "reason": "Lentil is suited to cooler Rabi weather and moderate water needs."},
    "pomegranate": {"season": ["Kharif", "Rabi", "Zaid"], "soil": ["Loamy", "Sandy", "Black"],
                    "water": ["Low", "Medium"], "temp": (20, 35), "humidity": (35, 70), "rainfall": (40, 120),
                    "ph": (5.5, 7.5), "reason": "Pomegranate can suit warm, relatively dry conditions with well-drained soil."},
    "banana": {"season": ["Kharif", "Zaid"], "soil": ["Alluvial", "Loamy", "Clay"], "water": ["High", "Medium"],
               "temp": (20, 35), "humidity": (60, 90), "rainfall": (100, 250), "ph": (5.5, 7.5),
               "reason": "Banana generally prefers warm, humid conditions and regular water."},
    "mango": {"season": ["Kharif", "Zaid"], "soil": ["Loamy", "Alluvial", "Red"], "water": ["Medium", "Low"],
              "temp": (24, 35), "humidity": (45, 80), "rainfall": (60, 200), "ph": (5.5, 7.5),
              "reason": "Mango is compatible with warm conditions and well-drained soil."},
    "grapes": {"season": ["Rabi", "Zaid"], "soil": ["Loamy", "Sandy"], "water": ["Medium", "Low"],
               "temp": (15, 32), "humidity": (40, 75), "rainfall": (50, 100), "ph": (5.5, 7.5),
               "reason": "Grapes generally prefer warm conditions, drainage and controlled moisture."},
    "watermelon": {"season": ["Zaid"], "soil": ["Sandy", "Loamy"], "water": ["Medium", "High"],
                   "temp": (22, 35), "humidity": (40, 75), "rainfall": (40, 100), "ph": (5.5, 7.5),
                   "reason": "Watermelon fits warm Zaid conditions and needs adequate moisture during growth."},
    "muskmelon": {"season": ["Zaid"], "soil": ["Sandy", "Loamy"], "water": ["Medium", "High"],
                  "temp": (22, 35), "humidity": (40, 75), "rainfall": (40, 100), "ph": (6.0, 7.5),
                  "reason": "Muskmelon is generally associated with warm conditions and well-drained soil."},
    "apple": {"season": ["Rabi"], "soil": ["Loamy", "Sandy"], "water": ["Medium", "High"],
              "temp": (8, 24), "humidity": (50, 80), "rainfall": (80, 180), "ph": (5.5, 7.0),
              "reason": "Apple is more compatible with cooler growing conditions and suitable moisture."},
    "orange": {"season": ["Kharif", "Rabi"], "soil": ["Loamy", "Alluvial", "Red"], "water": ["Medium", "High"],
               "temp": (18, 32), "humidity": (50, 80), "rainfall": (70, 180), "ph": (5.5, 7.5),
               "reason": "Orange can fit warm conditions with adequate moisture and well-drained soil."},
    "papaya": {"season": ["Kharif", "Zaid"], "soil": ["Loamy", "Alluvial", "Sandy"], "water": ["Medium", "High"],
               "temp": (21, 35), "humidity": (55, 85), "rainfall": (100, 200), "ph": (6.0, 7.5),
               "reason": "Papaya generally prefers warm conditions and regular moisture with drainage."},
    "coconut": {"season": ["Kharif", "Zaid"], "soil": ["Sandy", "Loamy", "Laterite"], "water": ["High", "Medium"],
                "temp": (22, 35), "humidity": (65, 95), "rainfall": (100, 300), "ph": (5.5, 8.0),
                "reason": "Coconut is compatible with warm humid conditions and adequate water."},
    "cotton": {"season": ["Kharif"], "soil": ["Black", "Loamy", "Red"], "water": ["Medium", "Low"],
               "temp": (21, 35), "humidity": (40, 80), "rainfall": (60, 150), "ph": (5.5, 8.0),
               "reason": "Cotton fits warm Kharif conditions and is commonly associated with black soils."},
    "jute": {"season": ["Kharif"], "soil": ["Alluvial", "Loamy", "Clay"], "water": ["High", "Medium"],
             "temp": (24, 35), "humidity": (65, 95), "rainfall": (150, 300), "ph": (5.5, 7.5),
             "reason": "Jute generally fits warm, humid conditions with ample rainfall and moisture."},
    "coffee": {"season": ["Kharif"], "soil": ["Red", "Loamy", "Laterite"], "water": ["Medium", "High"],
               "temp": (15, 30), "humidity": (60, 90), "rainfall": (120, 250), "ph": (5.0, 7.0),
               "reason": "Coffee is compatible with warm humid conditions, rainfall and acidic-to-neutral soil."},
}

DISPLAY_CROPS = {
    "rice": "Rice", "maize": "Maize", "chickpea": "Chickpea", "kidney beans": "Kidney Beans",
    "pigeon peas": "Pigeon Peas", "moth beans": "Moth Beans", "mung bean": "Mung Bean",
    "black gram": "Black Gram", "lentil": "Lentil", "pomegranate": "Pomegranate",
    "banana": "Banana", "mango": "Mango", "grapes": "Grapes", "watermelon": "Watermelon",
    "muskmelon": "Muskmelon", "apple": "Apple", "orange": "Orange", "papaya": "Papaya",
    "coconut": "Coconut", "cotton": "Cotton", "jute": "Jute", "coffee": "Coffee"
}

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            n REAL NOT NULL, p REAL NOT NULL, k REAL NOT NULL,
            temperature REAL NOT NULL, humidity REAL NOT NULL, rainfall REAL NOT NULL, ph REAL NOT NULL,
            soil TEXT NOT NULL, water TEXT NOT NULL, season TEXT NOT NULL,
            recommended_crop TEXT NOT NULL,
            health TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )""")
        conn.commit()

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception:
        return None

def numeric_score(value, low, high):
    if low <= value <= high:
        return 1.0
    distance = (low - value) if value < low else (value - high)
    span = max(high - low, 1.0)
    return max(0.0, 1.0 - distance / (span * 1.5))

def compatibility(crop_key, data):
    info = CROP_INFO[crop_key]
    scores = [
        numeric_score(data["temperature"], *info["temp"]),
        numeric_score(data["humidity"], *info["humidity"]),
        numeric_score(data["rainfall"], *info["rainfall"]),
        numeric_score(data["ph"], *info["ph"]),
        1.0 if data["soil"] in info["soil"] else 0.45,
        1.0 if data["water"] in info["water"] else 0.55,
        1.0 if data["season"] in info["season"] else 0.40,
    ]
    return sum(scores) / len(scores)

def health_status(data, crop_key):
    info = CROP_INFO[crop_key]
    checks = [
        numeric_score(data["temperature"], *info["temp"]),
        numeric_score(data["humidity"], *info["humidity"]),
        numeric_score(data["rainfall"], *info["rainfall"]),
        numeric_score(data["ph"], *info["ph"]),
        1.0 if data["soil"] in info["soil"] else 0.5,
        1.0 if data["water"] in info["water"] else 0.5,
    ]
    score = sum(checks) / len(checks)
    if score >= 0.88:
        return "Excellent", score
    if score >= 0.72:
        return "Good", score
    if score >= 0.52:
        return "Moderate", score
    return "Needs Attention", score

def reasons(data, crop_key):
    info = CROP_INFO[crop_key]
    items = []
    if info["temp"][0] <= data["temperature"] <= info["temp"][1]:
        items.append("Temperature is within the project compatibility range.")
    else:
        items.append("Temperature is outside the preferred project compatibility range.")
    if info["rainfall"][0] <= data["rainfall"] <= info["rainfall"][1]:
        items.append("Rainfall is compatible with this crop profile.")
    else:
        items.append("Rainfall is outside the preferred project compatibility range.")
    if info["ph"][0] <= data["ph"] <= info["ph"][1]:
        items.append("pH is within the project compatibility range.")
    else:
        items.append("pH is outside the preferred project compatibility range.")
    if data["soil"] in info["soil"]:
        items.append("Selected soil type is compatible with this project profile.")
    if data["water"] in info["water"]:
        items.append("Water availability matches this project profile.")
    if data["season"] in info["season"]:
        items.append(f"{data['season']} is listed as a suitable season in this project profile.")
    items.append(info["reason"])
    return items[:5]

def validate_form(form):
    fields = ["n", "p", "k", "temperature", "humidity", "rainfall", "ph", "soil", "water", "season"]
    if any(not str(form.get(x, "")).strip() for x in fields):
        return None, "Please fill in every prediction field."
    try:
        data = {
            "N": float(form["n"]), "P": float(form["p"]), "K": float(form["k"]),
            "temperature": float(form["temperature"]), "humidity": float(form["humidity"]),
            "rainfall": float(form["rainfall"]), "ph": float(form["ph"]),
            "soil": form["soil"], "water": form["water"], "season": form["season"]
        }
    except ValueError:
        return None, "Please enter valid numeric values."
    ranges = {
        "N": (0, 200), "P": (0, 200), "K": (0, 250),
        "temperature": (-10, 60), "humidity": (0, 100), "rainfall": (0, 500),
        "ph": (0, 14)
    }
    for key, (low, high) in ranges.items():
        if not low <= data[key] <= high:
            return None, f"{key} must be between {low} and {high}."
    if data["soil"] not in {"Alluvial", "Black", "Red", "Loamy", "Sandy", "Clay", "Laterite"}:
        return None, "Please select a valid soil type."
    if data["water"] not in {"Low", "Medium", "High"}:
        return None, "Please select a valid water availability."
    if data["season"] not in {"Kharif", "Rabi", "Zaid"}:
        return None, "Please select a valid season."
    return data, None

def predict_crop(data):
    model = load_model()
    ml_scores = {}
    if model is not None:
        try:
            x = np.array([[data[f] for f in ML_FEATURES]], dtype=float)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                classes = model.classes_
                ml_scores = {str(c).lower(): float(p) for c, p in zip(classes, probs)}
            else:
                pred = str(model.predict(x)[0]).lower()
                ml_scores[pred] = 1.0
        except Exception:
            ml_scores = {}

    ranked = []
    for crop_key in CROP_INFO:
        comp = compatibility(crop_key, data)
        ml = ml_scores.get(crop_key, 0.0)
        # ML contributes 75% when available; transparent compatibility layer contributes 25%.
        combined = (0.75 * ml + 0.25 * comp) if ml_scores else comp
        ranked.append((crop_key, combined, ml, comp))
    ranked.sort(key=lambda x: x[1], reverse=True)

    top = ranked[:3]
    best = top[0]
    # If model probabilities exist, confidence is the actual model probability.
    confidence = best[2] * 100 if ml_scores else best[1] * 100
    confidence = max(0.0, min(99.0, confidence))

    return {
        "best_key": best[0],
        "crop": DISPLAY_CROPS[best[0]],
        "confidence": round(confidence, 1),
        "alternatives": [
            {"crop": DISPLAY_CROPS[x[0]], "score": round(x[1] * 100, 1), "ml_used": bool(ml_scores)}
            for x in top[1:]
        ],
        "health": health_status(data, best[0])[0],
        "health_score": round(health_status(data, best[0])[1] * 100, 1),
        "season": ", ".join(CROP_INFO[best[0]]["season"]),
        "reasons": reasons(data, best[0]),
        "model_used": bool(ml_scores)
    }

@app.context_processor
def inject_globals():
    return {"logged_in_name": session.get("name")}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not name or not phone or not password or not confirm:
            flash("All fields are required.", "danger")
        elif not phone.isdigit() or len(phone) != 10 or phone[0] not in "6789":
            flash("Enter a valid 10-digit Indian mobile number.", "danger")
        elif len(password) < 6:
            flash("Password must contain at least 6 characters.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        else:
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO users(name, phone, password_hash, created_at) VALUES (?, ?, ?, ?)",
                        (name, phone, generate_password_hash(password), datetime.now().isoformat(timespec="seconds"))
                    )
                    conn.commit()
                flash("Registration successful. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("This phone number is already registered.", "danger")
            except Exception:
                flash("Registration could not be completed. Please try again.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            return redirect(url_for("dashboard"))
        flash("Invalid phone number or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM predictions WHERE user_id = ?", (session["user_id"],)).fetchone()["c"]
        last = conn.execute("SELECT recommended_crop, created_at FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                            (session["user_id"],)).fetchone()
    return render_template("dashboard.html", count=count, last=last)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        data, error = validate_form(request.form)
        if error:
            flash(error, "danger")
            return render_template("predict.html", form=request.form), 400
        result = predict_crop(data)
        try:
            with get_db() as conn:
                conn.execute("""INSERT INTO predictions
                    (user_id, created_at, n, p, k, temperature, humidity, rainfall, ph, soil, water, season, recommended_crop, health)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (session["user_id"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     data["N"], data["P"], data["K"], data["temperature"], data["humidity"], data["rainfall"], data["ph"],
                     data["soil"], data["water"], data["season"], result["crop"], result["health"]))
                conn.commit()
        except Exception:
            flash("Prediction was generated, but history could not be saved.", "warning")
        return render_template("results.html", data=data, result=result)
    return render_template("predict.html", form={})

@app.route("/history")
@login_required
def history():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC", (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows)

@app.post("/history/delete/<int:prediction_id>")
@login_required
def delete_prediction(prediction_id):
    with get_db() as conn:
        conn.execute("DELETE FROM predictions WHERE id = ? AND user_id = ?", (prediction_id, session["user_id"]))
        conn.commit()
    flash("Prediction deleted.", "success")
    return redirect(url_for("history"))

@app.post("/history/clear")
@login_required
def clear_history():
    with get_db() as conn:
        conn.execute("DELETE FROM predictions WHERE user_id = ?", (session["user_id"],))
        conn.commit()
    flash("Prediction history cleared.", "success")
    return redirect(url_for("history"))

@app.route("/health")
@login_required
def health():
    with get_db() as conn:
        last = conn.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                            (session["user_id"],)).fetchone()
    return render_template("health.html", last=last)

@app.route("/about")
def about():
    return render_template("about.html")

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(error):
    return render_template("500.html"), 500

init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
