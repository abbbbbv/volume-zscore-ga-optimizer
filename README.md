# Volume Z-Score GA Optimizer

## 📌 Description

Genetic algorithm-powered optimizer for backtesting a Binance Futures trading strategy based on Z-score volume surges and order flow analysis. This tool downloads historical data, detects statistically significant buy/sell pressure using volume anomalies, and evolves the best take-profit and stop-loss values using DEAP and backtesting.py.

---

## 🔧 Features

- 🔍 Z-score detection of volume spikes and order intent (BUY/SELL)
- 📊 Strategy execution with fixed TP/SL using backtesting.py
- 🧬 Evolutionary optimization using genetic algorithms (DEAP)
- ⚡ Parallel symbol optimization with multiprocessing
- ✅ Clean output with best strategy stats per symbol

---
install deap , backtesting.
