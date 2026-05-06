import random
import json
import pickle
import numpy as np
import nltk
import sqlite3

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# ---------------- NLTK SETUP ---------------- #
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------- DATABASE SETUP ---------------- #
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            fitness_goal TEXT,
            username TEXT UNIQUE,
            password TEXT,
            weight REAL,
            height REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workouts (
            workout_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            exercise_type TEXT,
            duration INTEGER,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def create_default_user():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", ("admin",))
    user = cursor.fetchone()
    if not user:
        cursor.execute("""
            INSERT INTO users (name, age, gender, fitness_goal, username, password)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("Admin", 25, "male", "General Fitness", "admin", "1234"))
        conn.commit()
    conn.close()

# ---------------- LOAD AI FILES ---------------- #
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

# ---------------- NLP FUNCTIONS ---------------- #
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words), dtype=int)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return bag

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.expand_dims(bow, axis=0), verbose=0)[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints):
    if not ints:
        return "Sorry, I didn’t understand."
    tag = ints[0]["intent"]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn’t understand."

# ---------------- WEEKLY PLAN ---------------- #
def generate_weekly_plan(age, gender, level):
    level = level.lower()

    intensity = {
        "easy": "Light",
        "medium": "Moderate",
        "hard": "High"
    }.get(level, "Moderate")

    text = f"""
📅 WEEKLY WORKOUT PLAN ({intensity} Level)

Monday: Chest + Triceps
Tuesday: Back + Biceps
Wednesday: Cardio
Thursday: Legs
Friday: Shoulders
Saturday: Abs + Core
Sunday: Rest

👤 Age: {age}
🔥 Level: {intensity}

Stay consistent 💪
"""
    return {"text": text, "videos": []}


# ---------------- WORKOUT & DIET FUNCTIONS ---------------- #
def generate_workout_plan(workout, age, gender, level):
    """
    Generates a weekly workout plan based on age, gender, and level.
    Returns a dictionary with 'text' and 'videos'.
    """

    age = int(age)
    gender = gender.lower()
    level = level.lower()

    # -------- AGE GROUP -------- #
    if age <= 18:
        age_group = "teen"
    elif age <= 30:
        age_group = "young"
    elif age <= 45:
        age_group = "adult"
    else:
        age_group = "senior"

    # -------- VIDEO LINKS -------- #
    exercise_videos = {
        "push-ups": "https://www.youtube.com/watch?v=IODxDxX7oi4",
        "bench press": "https://www.youtube.com/watch?v=rT7DgCr-3pg",
        "squats": "https://www.youtube.com/watch?v=aclHkVaku9U",
        "deadlift": "https://www.youtube.com/watch?v=op9kVnSso6Q",
        "lunges": "https://www.youtube.com/watch?v=QOVaHwm-Q6U",
        "plank": "https://www.youtube.com/watch?v=pSHjTRCQxIw",
        "bicep curl": "https://www.youtube.com/watch?v=ykJmrZ5v0Oo",
        "tricep": "https://www.youtube.com/watch?v=2-LAMcpzODU",
        "lat pulldown": "https://www.youtube.com/watch?v=CAwf7n6Luuc"
    }

    workouts = {
        # ================= MALE ================= #
        "male": {
            "young": {
                "easy": {
                    "chest": ["Push-ups – 3x15", "DB Press – 3x12", "Cable Fly – 3x15"],
                    "leg": ["Squats – 3x20", "Lunges – 3x15", "Calf Raises – 4x20"],
                    "back": ["Lat Pulldown – 3x12", "Seated Row – 3x12"],
                    "shoulder": ["DB Press – 3x12", "Lateral Raise – 3x15"],
                    "arm": ["Bicep Curl – 3x12", "Tricep Pushdown – 3x12"],
                    "abs": ["Crunches – 3x25", "Plank – 3x40 sec"]
                },
                "medium": {
                    "chest": ["Bench Press – 4x8", "Incline DB – 4x10", "Dips – 3x12"],
                    "leg": ["Barbell Squat – 4x8", "Leg Press – 4x10", "RDL – 3x10"],
                    "back": ["Pull-ups – 4x8", "Barbell Row – 4x10"],
                    "shoulder": ["Overhead Press – 4x8", "Arnold Press – 3x10"],
                    "arm": ["Barbell Curl – 4x10", "Skull Crusher – 3x10"],
                    "abs": ["Hanging Leg Raise – 3x15", "Cable Crunch – 3x15"]
                },
                "hard": {
                    "chest": ["Bench Press – 5x5 Heavy", "Incline Press – 4x6", "Weighted Dips – 4x8", "100 Push-ups"],
                    "leg": ["Heavy Squat – 5x5", "Deadlift – 4x6", "Leg Press Dropset", "Walking Lunges – 3x20"],
                    "back": ["Deadlift – 5x5", "Weighted Pull-ups – 4x6", "T-Bar Row – 4x8"],
                    "shoulder": ["Military Press – 5x5", "Lateral Raise Dropset", "Shrugs – 4x15"],
                    "arm": ["Barbell Curl – 4x8", "Close Grip Bench – 4x8", "Dips – Failure"],
                    "abs": ["Weighted Crunch – 4x15", "Plank – 1 min x4"]
                }
            }
        },
        # ================= FEMALE ================= #
        "female": {
            "young": {
                "easy": {
                    "chest": ["Wall Push-ups – 3x12", "Light DB Press – 3x12"],
                    "leg": ["Bodyweight Squats – 3x15", "Glute Bridge – 3x15", "Donkey Kicks – 3x15"],
                    "back": ["Lat Pulldown – 3x12", "Seated Row – 3x12"],
                    "shoulder": ["Light DB Press – 3x12", "Lateral Raise – 3x15"],
                    "arm": ["Light Bicep Curl – 3x12", "Tricep Kickback – 3x12"],
                    "abs": ["Crunches – 3x15", "Plank – 3x30 sec"]
                },
                "medium": {
                    "chest": ["Push-ups – 3x12", "Dumbbell Press – 4x10", "Chest Fly – 3x15"],
                    "leg": ["Squats – 4x12", "Hip Thrust – 4x12", "Walking Lunges – 3x15", "Glute Kickbacks – 3x15"],
                    "back": ["Assisted Pull-ups – 3x8", "Seated Row – 4x12"],
                    "shoulder": ["Arnold Press – 3x10", "Lateral Raise – 4x15"],
                    "arm": ["Bicep Curl – 3x12", "Rope Pushdown – 3x12"],
                    "abs": ["Leg Raise – 3x15", "Plank – 3x40 sec"]
                },
                "hard": {
                    "chest": ["Incline DB Press – 4x8", "Cable Press – 4x10", "Push-ups to Failure"],
                    "leg": ["Heavy Squats – 4x8", "Hip Thrust – 4x10", "Bulgarian Split Squat – 3x12", "Leg Press – 3x12"],
                    "back": ["Pull-ups – 4x8", "Barbell Row – 4x10", "Lat Pulldown Dropset"],
                    "shoulder": ["Military Press – 4x8", "Lateral Raise Dropset", "Rear Delt Fly – 4x15"],
                    "arm": ["Barbell Curl – 4x10", "Tricep Dips – Failure"],
                    "abs": ["Weighted Crunch – 4x15", "Hanging Leg Raise – 3x12", "Plank – 1 min x3"]
                }
            }
        }
    }

        # -------- COPY YOUNG WORKOUTS TO OTHER AGE GROUPS -------- #
    for g in ["male", "female"]:
        for age_grp in ["teen", "adult", "senior"]:
            workouts[g][age_grp] = workouts[g]["young"]

    # -------- GET PLAN -------- #
    plan = workouts.get(gender, {}).get(age_group, {}).get(level, {}).get(workout)
    if not plan:
        return {"text": "Workout not available for this category.", "videos": []}

    if not plan:
        return {"text": "Workout not available for this category.", "videos": []}

    result = f"\n🔥 {level.upper()} LEVEL {workout.upper()} WORKOUT ({gender.capitalize()} | Age {age}) 🔥\n\n"
    for ex in plan:
        result += f"• {ex}\n"
    result += "\n⚠ Warm-up 5 mins\n⚠ Stretch compulsory\n💧 Stay hydrated\n🔥 Stay consistent!"

    video_list = []
    for ex in plan:
        exercise_name = ex.split("–")[0].strip().lower()
        for key, url in exercise_videos.items():
            if key in exercise_name and url not in video_list:
                video_list.append(url)

    return {"text": result, "videos": video_list}

def generate_diet_plan(age, gender):
    age = int(age)
    gender = gender.lower()
    if age <= 25:
        if gender == "male":
            return "🥗 DIET PLAN (18–25 | Male)\nBreakfast: Eggs + Oats + Banana\nLunch: Rice + Chicken/Fish + Vegetables\nSnack: Nuts / Peanut butter\nDinner: Chapati + Paneer/Eggs\nWater: 3–4 Litres"
        else:
            return "🥗 DIET PLAN (18–25 | Female)\nBreakfast: Oats + Fruits\nLunch: Rice + Dal + Vegetables\nSnack: Nuts / Sprouts\nDinner: Chapati + Paneer\nWater: 2.5–3 Litres"
    elif age <= 40:
        if gender == "male":
            return "🥗 DIET PLAN (26–40 | Male)\nBreakfast: Idli/Oats + Eggs\nLunch: Brown rice + Protein\nSnack: Fruits / Nuts\nDinner: Light veg food\nAvoid junk & sugar"
        else:
            return "🥗 DIET PLAN (26–40 | Female)\nBreakfast: Fruits + Oats\nLunch: Rice + Vegetables\nSnack: Green tea + Nuts\nDinner: Light homemade food"
    else:
        return "🥗 DIET PLAN (40+)\nBreakfast: Fruits + Warm water\nLunch: Soft cooked food\nDinner: Very light meal\nAvoid sugar & fried foods\nDaily walking compulsory"

# ---------------- FLASK ROUTES ---------------- #
@app.route("/")
def login_page():
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        goal = request.form.get("goal")
        username = request.form.get("username")
        password = request.form.get("password")
        weight = float(request.form.get("weight"))
        height = float(request.form.get("height"))

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO users (name, age, gender, fitness_goal, username, password, weight, height)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, age, gender, goal, username, password, weight, height))
            conn.commit()
            conn.close()
            return redirect(url_for("login_page"))
        except:
            conn.close()
            return "Username already exists"
    return render_template("register.html")

@app.route("/login", methods=["POST"])
def handle_login():
    username = request.form.get("username")
    password = request.form.get("password")
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        session["user"] = username
        session["user_id"] = user[0]
        session["age"] = user[2]
        session["gender"] = user[3]
        session["weight"] = user[7]
        session["height"] = user[8]
        return redirect("/home")
    return render_template("login.html", error="Invalid Username or Password")

@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")
    return render_template("index.html")

# ---------------- GET ROUTE ---------------- #
@app.route("/get", methods=["POST"])
def get_route():
    response = chatbot_response()
    if isinstance(response, dict):
        return jsonify(response)
    else:
        return jsonify({"text": response, "videos": []})
        
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

@app.route("/history")
def history():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT exercise_type, duration, date FROM workouts WHERE user_id=?", (session["user_id"],))
    history_data = cursor.fetchall()
    conn.close()
    return render_template("history.html", workouts=history_data)

# ---------------- CHATBOT ---------------- #
def chatbot_response():

    if "user" not in session:
        return {"text": "Please login first.", "videos": []}

    message = (request.form.get("msg") or "").strip()
    if not message:
        return {"text": "Please type something.", "videos": []}

    message = message.lower()

    if "step" not in session:
        session["step"] = None

    workout_keywords = ["chest", "leg", "arm", "shoulder", "back", "abs"]

    # ---------------- STEP 1 DETECTION ---------------- #
    if session.get("step") is None:

        if any(word in message for word in workout_keywords):
            for word in workout_keywords:
                if word in message:
                    session["selected_workout"] = word
                    break
            session["step"] = "ask_gender_workout"
            return {"text": "Are you Male or Female?", "videos": []}

        elif "diet" in message:
            session["step"] = "ask_gender_diet"
            return {"text": "Are you Male or Female for diet plan?", "videos": []}

        elif "bmi" in message:
            session["step"] = "ask_weight"
            return {"text": "Enter your weight in kg.", "videos": []}

        else:
            ints = predict_class(message)
            return {"text": get_response(ints), "videos": []}

    # ---------------- WORKOUT FLOW ---------------- #

    if session["step"] == "ask_gender_workout":
        if message in ["male", "female"]:
            session["gender"] = message
            session["step"] = "ask_age_workout"
            return {"text": "Enter your age.", "videos": []}
        return {"text": "Type Male or Female.", "videos": []}

    if session["step"] == "ask_age_workout":
        if message.isdigit():
            session["age"] = int(message)
            session["step"] = "ask_level_workout"
            return {"text": "Select workout level: Easy / Medium / Hard", "videos": []}
        return {"text": "Enter a valid age.", "videos": []}

    if session["step"] == "ask_level_workout":
        if message in ["easy", "medium", "hard"]:

            level = message
            age = session.get("age")
            gender = session.get("gender")
            workout = session.get("selected_workout")

            session["step"] = None

            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username=?", (session["user"],))
            user = cursor.fetchone()

            if user:
                cursor.execute(
                    "INSERT INTO workouts (user_id, exercise_type, duration) VALUES (?, ?, ?)",
                    (user[0], f"{workout}_{level}_workout", 30)
                )
                conn.commit()

            conn.close()

            return generate_workout_plan(workout, age, gender, level)

        return {"text": "Select Easy / Medium / Hard", "videos": []}

    # ---------------- BMI FLOW ---------------- #

    # Trigger BMI
    if message.lower() == "bmi":
        session["step"] = "ask_weight"
        return {"text": "Enter your weight in kg (example: 70 or 70kg)", "videos": []}


    # Ask Weight
    if session.get("step") == "ask_weight":
        try:
            weight = float(message.replace("kg", "").strip())
            session["weight"] = weight
            session["step"] = "ask_height"
            return {"text": "Enter your height in cm (example: 160)", "videos": []}
        except:
            return {"text": "Enter valid weight (example: 70 or 70kg).", "videos": []}


    # Ask Height
    if session.get("step") == "ask_height":
        try:
            height_cm = float(message.replace("cm", "").strip())

            if height_cm < 100 or height_cm > 250:
                return {"text": "Enter valid height in cm (example: 160)", "videos": []}

            height = height_cm / 100
            weight = session.get("weight")

            bmi = weight / (height * height)
            session["step"] = None

            # ---------------- BMI CATEGORY ---------------- #

            if bmi < 18.5:
                status = "Underweight"

                diet = """
                🥗 DETAILED DIET PLAN:
                • Morning: Banana + Peanut Butter + Milk
                • Breakfast: 4 Eggs / Paneer + 2 Chapati
                • Lunch: Rice + Chicken/Fish or Dal + Vegetables
                • Evening: Dry fruits + Protein Shake
                • Dinner: Chapati + Curry + Curd
                • Before Bed: Warm milk

                🔥 Daily Calories Target: 2500-3000 kcal
                """

                workout_plan = """
                🏋 WEEKLY WORKOUT PLAN:
                Monday – Chest + Triceps
                Tuesday – Back + Biceps
                Wednesday – Rest
                Thursday – Legs
                Friday – Shoulders
                Saturday – Full Body
                Sunday – Rest

                Focus: Heavy strength training (8-12 reps)
                """

            elif bmi < 25:
                status = "Normal"

                diet = """
                🥗 DETAILED DIET PLAN:
                • Morning: Lemon water
                • Breakfast: Oats + Eggs / Idli + Sambar
                • Lunch: Brown rice + Chicken/Dal + Vegetables
                • Evening: Fruits + Nuts
                • Dinner: Chapati + Veg curry
                • Drink 3-4L water daily

                🔥 Daily Calories Target: 2000-2400 kcal
                """

                workout_plan = """
                🏋 WEEKLY WORKOUT PLAN:
                Monday – Chest
                Tuesday – Back
                Wednesday – Cardio (30 min)
                Thursday – Legs
                Friday – Shoulders
                Saturday – HIIT
                Sunday – Rest

                Mix strength + cardio
                """

            elif bmi < 30:
                status = "Overweight"

                diet = """
                🥗 FAT LOSS DIET PLAN:
                • Morning: Warm water + Lemon
                • Breakfast: 2 Eggs + Oats
                • Lunch: Grilled chicken / Dal + Vegetables
                • Evening: Green tea
                • Dinner: Soup + Salad
                • Avoid: Sugar, Bakery, Soft drinks

    
                🔥 Daily Calories Target: 1500-1800 kcal
                """

                workout_plan = """
                🏃 WEEKLY FAT LOSS PLAN:
                Daily – 30-40 min Cardio
                3 Days – HIIT
                3 Days – Light strength training
                10,000 steps daily

                Goal: Fat burn + calorie deficit
                """

            else:
                status = "Obese"

                diet = """
                🥗 STRICT WEIGHT LOSS PLAN:
                • Morning: Lemon water
                • Breakfast: Boiled eggs / Sprouts
                • Lunch: Salad + Grilled protein
                • Evening: Green tea
                • Dinner: Soup only
                • No sugar / No fried food / No junk

                🔥 Daily Calories Target: 1200-1500 kcal
                """

                workout_plan = """
                🚶 BEGINNER FAT LOSS PLAN:
                Daily – 30 min Walking
                Light stretching
                Basic bodyweight exercises
                Gradually increase intensity

                Goal: Safe weight reduction
                """

            return f"""
    📊 Your BMI is {round(bmi,2)}
    🧍 Category: {status}

    {diet}

    {workout_plan}

    ⚠ Consistency is key. Follow for minimum 8 weeks.
    """

        except:
            return {"text": "Enter valid height in cm (example: 160)", "videos": []}

    # ---------------- DIET FLOW ---------------- #

    if session["step"] == "ask_gender_diet":
        if message in ["male", "female"]:
            session["gender"] = message
            session["step"] = "ask_age_diet"
            return {"text": "Enter your age for diet plan.", "videos": []}
        return {"text": "Type Male or Female.", "videos": []}

    if session["step"] == "ask_age_diet":
        if message.isdigit():

            age = int(message)
            gender = session.get("gender")
            session["step"] = None

            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username=?", (session["user"],))
            user = cursor.fetchone()

            if user:
                cursor.execute(
                    "INSERT INTO workouts (user_id, exercise_type, duration) VALUES (?, ?, ?)",
                    (user[0], "diet_plan_request", 5)
                )
                conn.commit()

            conn.close()

            return generate_diet_plan(age, gender)

        return {"text": "Enter valid age.", "videos": []}

    # ---------------- DEFAULT CHATBOT ---------------- #

    ints = predict_class(message)
    response = get_response(ints)

    return {"text": response, "videos": []}
# ---------------- RUN ---------------- #
init_db()
create_default_user()

if __name__ == "__main__":
    app.run(debug=True, port=5001)