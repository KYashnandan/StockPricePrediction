# Stock Volatility and Price Prediction using a Hybrid Approach

This project predicts **stock volatility** as the primary target using a **hybrid approach that combines deep learning (LSTM) and ensemble methods (Random Forest)**.  
Stock price prediction is included as a secondary output.  
Interactive **Plotly** visualizations provide insights into volatility trends and stock price movements, making the analysis intuitive and visually appealing.

---

## Project Overview

Stock market volatility is a key indicator of risk and market uncertainty.  
This project leverages both deep learning and traditional machine learning to forecast **next-day volatility** while also providing stock price predictions.  
Through systematic **data preprocessing**, **feature engineering**, and **model training**, it delivers a reliable framework for analyzing both volatility and price trends.

### Workflow Summary
- Collect and clean historical stock data
- Generate technical indicators such as **RSI, Moving Averages, Volatility, ATR, and Bollinger Bands**
- Prepare datasets for both **LSTM** and **Random Forest** models
- Train and evaluate models to forecast **stock volatility** (primary) and **stock price** (secondary)
- Visualize actual and predicted trends using interactive **Plotly** charts

---

## Models Implemented

| Model | Dataset Suitability | Description |
|-------|-------------------|-------------|
| LSTM (Long Short-Term Memory) | Large sequential datasets | Captures temporal dependencies and long-term relationships in stock volatility and price movement data. |
| Random Forest Regressor | Small to medium datasets | Provides ensemble-based predictions that are stable, interpretable, and suitable for smaller datasets. |

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
- Enhanced visualization for volatility patterns and technical indicators

---

## License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute the code with appropriate attribution.

---

## About

This project demonstrates an end-to-end stock volatility prediction pipeline using a hybrid approach combining deep learning and ensemble methods. Stock price prediction is included as a secondary output. 
It is suitable for learning stock market modeling, feature engineering, and interactive visualization using Python and Plotly.



