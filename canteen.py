import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

CSV_FILE = "canteen.csv"
FEATURES = ["temperature", "is_weekend", "is_exam_day", "is_raining"]
model = None
train_min = None
train_max = None
residual_std = None


def ask(question, allowed=None, minimum=None):
    while True:
        text = console.input(f"[cyan]{question}[/cyan]").strip()
        try:
            value = float(text)
        except ValueError:
            console.print("  [yellow]Please type a number.[/yellow]")
            continue
        if minimum is not None and value < minimum:
            console.print(f"  [yellow]Please type a number that is {minimum} or more.[/yellow]")
            continue
        if allowed is not None and value not in allowed:
            console.print(f"  [yellow]Please type one of: {allowed}[/yellow]")
            continue
        return value


def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=FEATURES + ["plates_sold"])


def add_day():
    console.print(Panel("Add a day of data", style="bold green"))
    row = {
        "temperature": ask("Temperature that day (C): "),
        "is_weekend": ask("Weekend? (1 = yes, 0 = no): ", [0, 1]),
        "is_exam_day": ask("Exam day? (1 = yes, 0 = no): ", [0, 1]),
        "is_raining": ask("Raining? (1 = yes, 0 = no): ", [0, 1]),
        "plates_sold": ask("Plates sold that day: ", minimum=0),
    }
    df = pd.concat([load_data(), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    console.print(f"[green]Saved! You now have {len(df)} days of data.[/green]")


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
    console.print(Panel(
        f"Loaded {n} FAKE demo days (replace with real data later).\n"
        "This fake data was generated from a straight-line formula, so it will\n"
        "always look easier to predict than real sales. Treat any R2 you see on\n"
        "demo data as a sanity check, not proof the model will work on real numbers.",
        title="Demo data loaded", style="yellow",
    ))


def train():
    global model, train_min, train_max, residual_std
    df = load_data().dropna()
    if len(df) < 8:
        console.print(f"[yellow]You only have {len(df)} days. Add at least 8 (30+ is better).[/yellow]")
        return False

    X, y = df[FEATURES], df["plates_sold"]
    train_min, train_max = X.min(), X.max()

    n_splits = min(5, max(2, len(df) // 4))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    model = PoissonRegressor(alpha=1e-3, max_iter=1000)

    cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    cv_preds = cross_val_predict(model, X, y, cv=kf)
    residual_std = float(np.std(y.values - cv_preds))

    model.fit(X, y)

    console.print(Panel(
        f"Trained on {len(df)} days using {n_splits}-fold cross-validation.\n"
        f"R2: mean [bold]{cv_scores.mean():.2f}[/bold], "
        f"range {cv_scores.min():.2f} to {cv_scores.max():.2f}\n"
        f"Typical prediction error: about {residual_std:.0f} plates",
        title="Training complete", style="green" if cv_scores.mean() > 0.5 else "yellow",
    ))
    if len(df) < 20:
        console.print("[yellow]With under 20 days, treat this R2 as a rough signal - add more days.[/yellow]")

    table = Table(title="What the model learned")
    table.add_column("Factor", style="cyan")
    table.add_column("Effect on plates sold", justify="right")
    table.add_row("base sales", f"{np.exp(model.intercept_):.0f} plates")
    for name, c in zip(FEATURES, model.coef_):
        pct = (np.exp(c) - 1) * 100
        style = "green" if pct > 0 else "red"
        table.add_row(name, f"[{style}]{pct:+.1f}%[/{style}]")
    console.print(table)
    return True


def predict_tomorrow():
    if model is None and not train():
        return
    console.print(Panel("Predict tomorrow", style="bold green"))
    day = pd.DataFrame([[
        ask("Tomorrow's temperature (C): "),
        ask("Weekend? (1 = yes, 0 = no): ", [0, 1]),
        ask("Exam day? (1 = yes, 0 = no): ", [0, 1]),
        ask("Rain expected? (1 = yes, 0 = no): ", [0, 1]),
    ]], columns=FEATURES)

    out_of_range = False
    for name in FEATURES:
        v = day[name].iloc[0]
        if v < train_min[name] or v > train_max[name]:
            out_of_range = True
            console.print(f"  [yellow]WARNING: {name} = {v:g} is outside your data "
                           f"({train_min[name]:g} to {train_max[name]:g}). Prediction may be unreliable.[/yellow]")

    expected = model.predict(day)[0]
    buffer_plates = residual_std * (1.5 if out_of_range else 1.0)
    cook = expected + buffer_plates

    result = Text()
    result.append(f"Expected sales : about {expected:.0f} plates\n", style="bold")
    result.append(f"Cook           : about {cook:.0f} plates ", style="bold green")
    result.append(f"(includes a ~{buffer_plates:.0f}-plate safety buffer)")
    console.print(Panel(result, title="Tomorrow's forecast"))


def show_graphs():
    if model is None and not train():
        return
    df = load_data().dropna()
    X, y = df[FEATURES], df["plates_sold"]
    pred = model.predict(X)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))


    ax[0].scatter(y, pred, color="#2a9d8f", edgecolor="white", s=50, alpha=0.7)
    lo, hi = y.min(), y.max()
    ax[0].plot([lo, hi], [lo, hi], color="#e76f51", linewidth=2)
    ax[0].set_xlabel("Actual plates"); ax[0].set_ylabel("Predicted plates")
    ax[0].set_title("Actual vs predicted\n(closer to red line = better)")


    pct_effects = (np.exp(model.coef_) - 1) * 100
    colors = ["#2a9d8f" if c > 0 else "#e76f51" for c in pct_effects]
    ax[1].barh(FEATURES, pct_effects, color=colors)
    ax[1].set_title("Effect of each factor\n(% change in plates sold)")
    ax[1].axvline(0, color="black", linewidth=0.8)


    ax[2].scatter(df["temperature"], y, label="actual", alpha=0.6, color="#264653")
    order = np.argsort(df["temperature"].values)
    ax[2].plot(df["temperature"].values[order], np.array(pred)[order],
               color="#e76f51", linewidth=2, label="model trend")
    ax[2].set_xlabel("Temperature (C)"); ax[2].set_ylabel("Plates sold")
    ax[2].set_title("Temperature vs sales")
    ax[2].legend()

    plt.tight_layout()
    plt.savefig("canteen_graphs.png", dpi=150)
    console.print("[green]Graphs saved to canteen_graphs.png[/green]")
    plt.show()


def show_data():
    df = load_data()
    if len(df) == 0:
        console.print("[yellow]No data yet.[/yellow]")
        return
    table = Table(title=f"Your data ({len(df)} days)")
    for col in FEATURES + ["plates_sold"]:
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(*[f"{r[c]:g}" for c in FEATURES + ["plates_sold"]])
    console.print(table)


def main():
    global model
    console.print(Panel.fit("CANTEEN SALES PREDICTOR", style="bold white on blue"))
    menu_table = Table(show_header=False, box=None)
    menu_table.add_row("1", "Add a day of data (type it in)")
    menu_table.add_row("2", "Load 200 fake demo days")
    menu_table.add_row("3", "Train the model / see what it learned")
    menu_table.add_row("4", "Predict tomorrow")
    menu_table.add_row("5", "Show graphs")
    menu_table.add_row("6", "Show my data")
    menu_table.add_row("7", "Quit")

    while True:
        console.print(menu_table)
        try:
            choice = console.input("[bold cyan]Choose 1-7: [/bold cyan]").strip()
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
                show_graphs()
            elif choice == "6":
                show_data()
            elif choice == "7":
                console.print("[bold]Bye![/bold]"); break
            else:
                console.print("[yellow]Please choose 1 to 7.[/yellow]")
        except EOFError:
            break


if __name__ == "__main__":
    main()