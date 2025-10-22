# Stock Price Prediction using LSTM and Random Forest

This project predicts stock prices using two complementary machine learning approaches: **Long Short-Term Memory (LSTM)** networks and **Random Forest Regressors**.  
LSTM is applied to large sequential datasets where temporal dependencies are critical, while Random Forest is used for smaller datasets requiring faster training and interpretability.  
Interactive **Plotly** visualizations provide an intuitive understanding of model performance and market trends.

---

## Project Overview

Stock market prices are inherently volatile and influenced by numerous dynamic factors.  
This project combines deep learning and traditional machine learning to capture both short-term fluctuations and long-term patterns.  
Through systematic **data preprocessing**, **feature engineering**, and **model training**, it delivers a reliable framework for forecasting stock prices.

### Workflow Summary
- Collect and clean historical stock data
- Generate technical indicators such as RSI and Moving Averages
- Prepare datasets tailored for both LSTM and Random Forest models
- Train and evaluate models to forecast stock movements
- Visualize actual and predicted trends through interactive 3D Plotly graphs

---

## Models Implemented

| Model | Dataset Suitability | Description |
|-------|-------------------|-------------|
| LSTM (Long Short-Term Memory) | Large sequential datasets | Captures temporal dependencies and long-term relationships in stock movement data. |
| Random Forest Regressor | Small to medium datasets | Provides ensemble-based predictions that are stable, interpretable, and require minimal preprocessing. |

---

## Installation and Setup

1. **Clone the Repository**
```bash
git clone https://github.com/KYashnandan/StockPricePrediction.git
cd StockPricePrediction
```

2. **Install Required Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Prediction Model**
```bash
python stock_predictor.py
```

4. **Input Parameters**

When prompted, enter a valid stock ticker (e.g., `AAPL`, `TSLA`) and specify a date range.

---

## Project Structure

```
StockPricePrediction/
│
├── stock_predictor.py      # Main execution script
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## Future Improvements

- Integration of additional deep learning architectures (e.g., GRU)
- Real-time data fetching from financial APIs
- Model explainability using SHAP or LIME
- Enhanced visualization for more technical indicators

---

## License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute the code with appropriate attribution.

---

## About

This project demonstrates an end-to-end stock price prediction pipeline using both deep learning and traditional machine learning methods.  
It is suitable for learning stock market modeling, feature engineering, and interactive visualization using Python and Plotly.


