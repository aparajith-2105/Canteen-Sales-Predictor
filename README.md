# Canteen Sales Predictor

A terminal app that predicts how many plates a canteen will sell tomorrow, using linear regression. You type in past days of data, train the model, and it tells you how many plates to cook.

## What it does

- Stores daily records (temperature, weekend, exam day, rain, plates sold) in `canteen.csv`
- Trains a linear regression model with K-fold cross-validation
- Shows how much each factor changes sales, in plates
- Predicts tomorrow's sales and suggests how many plates to cook, with a safety buffer
- Warns you when tomorrow's inputs fall outside the range of your data

## Requirements

- Python 3.9 or newer
- Libraries: `numpy`, `pandas`, `scikit-learn`, `rich`

Install them with:

```bash
pip install numpy pandas scikit-learn rich
```

## How to run

```bash
python canteen_predictor_linear.py
```

The app opens a menu:

| Option | What it does |
|---|---|
| 1 | Add one day of data by typing it in |
| 2 | Load 200 fake demo days (for testing) |
| 3 | Train the model and see what it learned |
| 4 | Predict tomorrow |
| 5 | Show all saved data |
| 6 | Quit |

## Quick start

1. Choose **2** to load demo data, or choose **1** repeatedly to enter real days.
2. Choose **3** to train and see the effect of each factor.
3. Choose **4**, answer the four questions about tomorrow, and read the forecast.

## Input columns

| Column | Meaning | Values |
|---|---|---|
| `temperature` | Temperature that day | Degrees Celsius |
| `is_weekend` | Weekend or not | 1 = yes, 0 = no |
| `is_exam_day` | Exam day or not | 1 = yes, 0 = no |
| `is_raining` | Rain or not | 1 = yes, 0 = no |
| `plates_sold` | Plates sold that day (target) | 0 or more |

## How the prediction works

- **Model:** `LinearRegression`. Sales = base sales + (effect of each factor). Each factor adds or subtracts a fixed number of plates.
- **Validation:** K-fold cross-validation (up to 5 folds). The app reports the mean R² and its range. Closer to 1 is better.
- **Cook amount:** expected sales plus a safety buffer equal to the typical prediction error. The buffer is increased by 1.5 times if any input is outside your training data.
- **No negatives:** predictions are clipped at 0 plates.

## Tips and limits

- You need at least 8 days to train, and 30 or more days for a reliable result. Under 20 days, treat the R² as a rough signal.
- The demo data is generated from a straight-line formula, so it looks easier to predict than real sales. Use it only to test the app.
- **Option 2 overwrites `canteen.csv`.** Back up your real data before loading the demo.
- The model treats all days as independent. It does not know about trends over time, festivals, or menu changes unless you add them as columns.
- Linear regression suits large daily totals (roughly 200+ plates). For small per-item counts, a Poisson regression version is a better fit.

## Files

- `canteen_predictor_linear.py`: the program
- `canteen.csv`: created automatically when you add data or load the demo
