# 🍽️ Canteen Sales Predictor

A machine learning-based canteen demand prediction system that estimates the number of plates likely to be sold on a given day. It helps canteen operators decide **how much food to prepare**, reducing both food wastage and shortages.

## 🎯 Problem

Canteens often have difficulty estimating daily food demand. Sales can change depending on factors such as **temperature, weekends, exams, and rainfall**.

This project uses these factors along with historical sales data to predict the expected number of plates sold.

## 💡 Solution

The system uses **Poisson Regression** to model plate sales as count data.

The model takes four factors as input:

* 🌡️ **Temperature**
* 📅 **Weekend status**
* 📝 **Exam day status**
* 🌧️ **Rain status**

It then predicts the expected number of plates to be sold and adds a **safety buffer based on the model's prediction error** to recommend how many plates should be prepared.

## 🤖 Machine Learning

* **Model:** Poisson Regression
* **Validation:** K-Fold Cross-Validation
* **Evaluation:** R² score and prediction error
* **Data storage:** CSV
* **Minimum training data:** 8 days
* **Recommended:** 30+ days of real data

The application also checks whether prediction inputs are outside the range of the training data and warns the user when the prediction may be less reliable.

## 📊 Visualizations

The project generates three visualizations:

1. **Actual vs Predicted Sales** – compares real sales with model predictions.
2. **Feature Effects** – shows how each factor affects predicted sales.
3. **Temperature vs Sales** – displays the relationship between temperature and plate sales.

## ✨ Features

* Add real daily canteen sales data
* Automatically save data to `canteen.csv`
* Train and evaluate the ML model
* Predict the next day's sales
* Calculate a practical cooking recommendation
* Detect out-of-range prediction inputs
* Display learned feature effects
* Generate graphs for analysis
* Includes **200 synthetic demo records** for testing

## 🛠️ Technologies Used

**Python · NumPy · Pandas · Scikit-learn · Matplotlib · Rich**

## ▶️ Installation & Usage

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn rich
```

Run the application:

```bash
python canteen_sales_predictor.py
```

The interactive menu allows users to add data, train the model, make predictions, view graphs, and inspect the collected dataset.

## 📁 Project Files

```text
canteen_sales_predictor.py   # Main application
canteen.csv                  # Sales data
canteen_graphs.png           # Generated visualizations
README.md                    # Project documentation
```
## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/aparajith-2105/Canteen-Sales-Predictor.git
cd Canteen-Sales-Predictor
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn rich
```

### 3. Run the Application

```bash
python canteen.py
```

### 4. Try the Demo

For a quick demonstration, select the following options from the menu:

```text
2 → Load 200 fake demo days
3 → Train the model
4 → Predict tomorrow
5 → Show graphs
```

You can also choose **1** to add your own daily canteen data.

The application stores the collected data in `canteen.csv` and saves the generated graphs as `canteen_graphs.png`.



## ⚠️ Note

The included demo data is **synthetically generated** for testing the application. Real-world performance depends on collecting sufficient and reliable historical canteen data.

## 🚀 Future Scope

The project can be extended with additional factors such as **day-specific demand, menu type, holidays, special events, food prices, and historical weather data**, as well as a web or mobile interface.
