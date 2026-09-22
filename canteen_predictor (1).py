

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

CSV_FILE = "canteen.csv"
FEATURES = ["temperature", "is_weekend", "is_exam_day", "is_raining"]
BUFFER = 0.10          
model = None
train_min = None       
train_max = None       


def ask(question, allowed=None, minimum=None):
    while True:
        text = input(question).strip()
        try:
            value = float(text)
        except ValueError:
            print("  Please type a number.")
            continue
        if minimum is not None and value < minimum:
            print(f"  Please type a number that is {minimum} or more.")
            continue
        if allowed is not None and value not in allowed:
            print(f"  Please type one of: {allowed}")
            continue
        return value


def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=FEATURES + ["plates_sold"])


def add_day():
    print("\n--- Add a day of data ---")
    row = {
        "temperature": ask("Temperature that day (C): "),
        "is_weekend": ask("Weekend? (1 = yes, 0 = no): ", [0, 1]),
        "is_exam_day": ask("Exam day? (1 = yes, 0 = no): ", [0, 1]),
        "is_raining": ask("Raining? (1 = yes, 0 = no): ", [0, 1]),
        "plates_sold": ask("Plates sold that day: ", minimum=0),
    }
    df = pd.concat([load_data(), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved! You now have {len(df)} days of data.")


def load_demo():
    np.random.seed(1)
    n = 200
    df = pd.DataFrame({
        "temperature": np.random.randint(24, 40, n),
        "is_weekend": np.random.randint(0, 2, n),
        "is_exam_day": np.random.randint(0, 2, n),
        "is_raining": np.random.randint(0, 2, n),
    })
    df["plates_sold"] = (150 + 3 * df.temperature - 40 * df.is_weekend
                         - 25 * df.is_exam_day - 30 * df.is_raining
                         + np.random.normal(0, 8, n)).round()
    df["plates_sold"] = df["plates_sold"].clip(lower=0)
    df.to_csv(CSV_FILE, index=False)
    print(f"Loaded {n} FAKE demo days (replace with real data later).")

def train():
    global model, train_min, train_max
    df = load_data().dropna()
    if len(df) < 8:
        print(f"\nYou only have {len(df)} days. Add at least 8 (30+ is better).")
        return False
    X, y = df[FEATURES], df["plates_sold"]
    train_min, train_max = X.min(), X.max()
    model = LinearRegression()
    if len(df) >= 20:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)
        model.fit(Xtr, ytr)
        print(f"\nTrained on {len(Xtr)} days. R2 on {len(Xte)} unseen days: "
              f"{model.score(Xte, yte):.2f} (closer to 1 = better)")
    else:
        model.fit(X, y)
        print(f"\nTrained on {len(df)} days. R2: {model.score(X, y):.2f} "
              "(add 20+ days for a fair test)")
    print("\nWhat the model learned:")
    print(f"  base sales: {model.intercept_:.1f} plates")
    for name, c in zip(FEATURES, model.coef_):
        print(f"  {name}: {c:+.1f} plates")
    return True



def predict_tomorrow():
    if model is None and not train():
        return
    print("\n--- Predict tomorrow ---")
    day = pd.DataFrame([[
        ask("Tomorrow's temperature (C): "),
        ask("Weekend? (1 = yes, 0 = no): ", [0, 1]),
        ask("Exam day? (1 = yes, 0 = no): ", [0, 1]),
        ask("Rain expected? (1 = yes, 0 = no): ", [0, 1]),
    ]], columns=FEATURES)
    
    for name in FEATURES:
        v = day[name].iloc[0]
        if v < train_min[name] or v > train_max[name]:
            print(f"  WARNING: {name} = {v:g} is outside your data "
                  f"({train_min[name]:g} to {train_max[name]:g}). Prediction may be unreliable.")

    raw = model.predict(day)[0]
    expected = max(raw, 0)                    
    if raw < 0:
        print(f"\n  The model's raw answer was {raw:.0f}, which is impossible.")
        print("  This usually means too little data or a day very different from your data.")
        print("  Showing 0 instead. Add more days (30+) and retrain.")
    print(f"\nExpected sales : about {expected:.0f} plates")
    print(f"Cook           : about {expected * (1 + BUFFER):.0f} plates "
          f"(includes {int(BUFFER * 100)}% safety buffer)")


print("=== CANTEEN SALES PREDICTOR ===")
while True:
    print("\n1) Add a day of data (type it in)")
    print("2) Load 200 fake demo days")
    print("3) Train the model / see what it learned")
    print("4) Predict tomorrow")
    print("5) Show my data")
    print("6) Quit")
    try:
        choice = input("Choose 1-6: ").strip()
    except EOFError:
        break
    try:
        if choice == "1":
            add_day(); model = None         
        elif choice == "2":
            load_demo(); model = None
        elif choice == "3":
            train()
        elif choice == "4":
            predict_tomorrow()
        elif choice == "5":
            print(load_data().to_string())
        elif choice == "6":
            print("Bye!"); break
        else:
            print("Please choose 1 to 6.")
    except EOFError:
        break
