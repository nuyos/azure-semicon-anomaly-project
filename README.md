#  Azure-Based Real-Time Semiconductor Anomaly Detection Project

###  Overview
This project aims to build a **real-time defect (anomaly) detection pipeline** for semiconductor sensor data using **Microsoft Azure** cloud services.  
Incoming process data is streamed, analyzed, and visualized in near real-time to identify potential defects early in the production line.

---

##  Azure Architecture

| Component | Role | Description |
|------------|------|--------------|
| **Azure Blob Storage** | Data Lake | Stores raw and processed sensor data (CSV, JSON) |
| **Azure Stream Analytics** | Real-Time Processing | Ingests and filters live data streams |
| **Azure Machine Learning** | Model Training & Serving | Builds and deploys anomaly detection models |
| **Azure SQL Database** | Data Warehouse | Stores processed data and prediction results |
| **Azure Functions** | Automation | Triggers model inference when new data arrives |
| **Azure Web App (Streamlit)** | Visualization | Displays real-time dashboard for defect monitoring |

---

##  Tech Stack
- **Language:** Python 3.11  
- **Environment:** VS Code + Jupyter Notebook  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `azure-storage-blob`, `streamlit`, `sqlalchemy`, etc
- **Version Control:** Git + GitHub (branch-based workflow)

---

##  Repository Structure

```
semiconductor-realtime-anomaly-detection/
├── data/
│   ├── raw/              # Raw sensor data (ignored in Git)
│   └── processed/        # Cleaned / preprocessed data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
├── functions/            # Azure Functions code
├── webapp/               # Streamlit or Flask dashboard
├── docs/                 # Architecture diagrams, reports
├── requirements.txt
└── README.md
```
