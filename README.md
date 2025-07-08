
# Stock Price Trend Prediction Using LSTM and Streamlit

This project is a **Stock Price Prediction Web Application** built using **Long Short-Term Memory (LSTM)** neural networks and deployed with **Streamlit**.
It allows users to interactively visualize historical stock trends and predict future stock prices.

---

**Live Demo:** [Streamlit App](https://stock-price-prediction-hga4xr6ul7uzznxll83dti.streamlit.app/)

---

## Features

* Select any **Start Date** and **End Date** for historical stock data
* Input any **Stock Ticker Symbol** (e.g., AAPL, TSLA, MSFT)
* Visualizations include:

  * Closing Price vs Time
  * 100-Day Moving Average
  * 100 & 200-Day Moving Averages
  * Predicted vs Actual Prices
* LSTM model trained on historical stock prices
* Clean and intuitive user interface built with Streamlit

---

## Technologies Used

* Python
* Streamlit
* Keras and TensorFlow (LSTM)
* Scikit-learn
* Matplotlib and Pandas
* Yahoo Finance API (`yfinance`)

---

## How to Run Locally

1. **Clone the repository**:

   ```bash
   git clone https://github.com/GulshanB01/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

---
