import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
            console.print(
                f"  [yellow]Please type a number that is {minimum} or more.[/yellow]"
            )
            continue

        if allowed is not None and value not in allowed:
            console.print(
                f"  [yellow]Please type one of: {allowed}[/yellow]"
            )
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
        "is_weekend": ask(
            "Weekend? (1 = yes, 0 = no): ",
            [0, 1]
        ),
        "is_exam_day": ask(
            "Exam day? (1 = yes, 0 = no): ",
            [0, 1]
        ),
        "is_raining": ask(
            "Raining? (1 = yes, 0 = no): ",
            [0, 1]
        ),
        "plates_sold": ask(
            "Plates sold that day: ",
            minimum=0
        ),
    }

    df = pd.concat(
        [load_data(), pd.DataFrame([row])],
        ignore_index=True
    )

    df.to_csv(CSV_FILE, index=False)

    console.print(
        f"[green]Saved! You now have {len(df)} days of data.[/green]"
    )


def load_demo():
    np.random.seed(1)

    n = 200

    df = pd.DataFrame({
        "temperature": np.random.randint(24, 40, n),
        "is_weekend": np.random.randint(0, 2, n),
        "is_exam_day": np.random.randint(0, 2, n),
        "is_raining": np.random.randint(0, 2, n),
    })

    # Linear formula used to create demo sales data
    df["plates_sold"] = (
        150
        + 3 * df["temperature"]
        - 40 * df["is_weekend"]
        - 25 * df["is_exam_day"]
        - 30 * df["is_raining"]
        + np.random.normal(0, 8, n)
    ).round()

    df["plates_sold"] = df["plates_sold"].clip(lower=0)

    df.to_csv(CSV_FILE, index=False)

    console.print(
        Panel(
            f"Loaded {n} FAKE demo days.\n"
            "Replace this with real canteen data later.\n\n"
            "The demo data was generated using a linear formula, "
            "so Linear Regression should work well on it.",
            title="Demo data loaded",
            style="yellow",
        )
    )


def train():
    global model, train_min, train_max, residual_std

    df = load_data().dropna()

    if len(df) < 8:
        console.print(
            f"[yellow]You only have {len(df)} days. "
            "Add at least 8 days (30+ is better).[/yellow]"
        )
        return False

    X = df[FEATURES]
    y = df["plates_sold"]

    train_min = X.min()
    train_max = X.max()

    # Cross-validation
    n_splits = min(5, max(2, len(df) // 4))

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=1
    )

    # Linear Regression model
    model = LinearRegression()

    # R² score
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    # Cross-validation predictions
    cv_preds = cross_val_predict(
        model,
        X,
        y,
        cv=kf
    )

    # Prediction error
    residual_std = float(
        np.std(y.values - cv_preds)
    )

    # Train final model using all data
    model.fit(X, y)

    console.print(
        Panel(
            f"Trained on {len(df)} days using "
            f"{n_splits}-fold cross-validation.\n"
            f"R²: mean [bold]{cv_scores.mean():.2f}[/bold], "
            f"range {cv_scores.min():.2f} to "
            f"{cv_scores.max():.2f}\n"
            f"Typical prediction error: about "
            f"{residual_std:.0f} plates",
            title="Training complete",
            style="green" if cv_scores.mean() > 0.5 else "yellow",
        )
    )

    if len(df) < 20:
        console.print(
            "[yellow]With under 20 days, treat this R² "
            "as a rough signal - add more days.[/yellow]"
        )

    # Show what the model learned
    table = Table(title="What the model learned")

    table.add_column("Factor", style="cyan")
    table.add_column(
        "Effect on plates sold",
        justify="right"
    )

    # Intercept
    table.add_row(
        "Base sales",
        f"{model.intercept_:.1f} plates"
    )

    # Coefficients
    for name, coefficient in zip(
        FEATURES,
        model.coef_
    ):

        style = "green" if coefficient > 0 else "red"

        table.add_row(
            name,
            f"[{style}]{coefficient:+.1f} plates[/{style}]"
        )

    console.print(table)

    return True


def predict_tomorrow():

    if model is None:
        if not train():
            return

    console.print(
        Panel(
            "Predict tomorrow",
            style="bold green"
        )
    )

    day = pd.DataFrame(
        [[
            ask("Tomorrow's temperature (C): "),
            ask(
                "Weekend? (1 = yes, 0 = no): ",
                [0, 1]
            ),
            ask(
                "Exam day? (1 = yes, 0 = no): ",
                [0, 1]
            ),
            ask(
                "Rain expected? (1 = yes, 0 = no): ",
                [0, 1]
            ),
        ]],
        columns=FEATURES
    )

    # Check whether values are outside training range
    out_of_range = False

    for name in FEATURES:

        value = day[name].iloc[0]

        if (
            value < train_min[name]
            or value > train_max[name]
        ):

            out_of_range = True

            console.print(
                f"  [yellow]WARNING: {name} = {value:g} "
                f"is outside your data "
                f"({train_min[name]:g} to "
                f"{train_max[name]:g}). "
                "Prediction may be unreliable.[/yellow]"
            )

    # Linear regression prediction
    expected = model.predict(day)[0]

    # Sales cannot be negative
    expected = max(0.0, expected)

    # Safety buffer
    buffer_plates = residual_std * (
        1.5 if out_of_range else 1.0
    )

    cook = expected + buffer_plates

    result = Text()

    result.append(
        f"Expected sales : about {expected:.0f} plates\n",
        style="bold"
    )

    result.append(
        f"Cook           : about {cook:.0f} plates ",
        style="bold green"
    )

    result.append(
        f"(includes a ~{buffer_plates:.0f}-plate safety buffer)"
    )

    console.print(
        Panel(
            result,
            title="Tomorrow's forecast"
        )
    )


def show_data():

    df = load_data()

    if len(df) == 0:
        console.print(
            "[yellow]No data yet.[/yellow]"
        )
        return

    table = Table(
        title=f"Your data ({len(df)} days)"
    )

    for column in FEATURES + ["plates_sold"]:
        table.add_column(column)

    for _, row in df.iterrows():

        table.add_row(
            *[
                f"{row[column]:g}"
                for column in FEATURES + ["plates_sold"]
            ]
        )

    console.print(table)


def main():

    global model

    console.print(
        Panel.fit(
            "CANTEEN SALES PREDICTOR",
            style="bold white on blue"
        )
    )

    menu_table = Table(
        show_header=False,
        box=None
    )

    menu_table.add_row(
        "1",
        "Add a day of data"
    )

    menu_table.add_row(
        "2",
        "Load 200 fake demo days"
    )

    menu_table.add_row(
        "3",
        "Train the Linear Regression model"
    )

    menu_table.add_row(
        "4",
        "Predict tomorrow"
    )

    menu_table.add_row(
        "5",
        "Show my data"
    )

    menu_table.add_row(
        "6",
        "Quit"
    )

    while True:

        console.print(menu_table)

        try:
            choice = console.input(
                "[bold cyan]Choose 1-6: [/bold cyan]"
            ).strip()

        except EOFError:
            break

        try:

            if choice == "1":

                add_day()
                model = None

            elif choice == "2":

                load_demo()
                model = None

            elif choice == "3":

                train()

            elif choice == "4":

                predict_tomorrow()

            elif choice == "5":

                show_data()

            elif choice == "6":

                console.print("[bold]Bye![/bold]")
                break

            else:

                console.print(
                    "[yellow]Please choose 1 to 6.[/yellow]"
                )

        except EOFError:
            break


# Correct Python main check
if __name__ == "__main__":
    main()
