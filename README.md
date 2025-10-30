# 📦 Retail Basket Analysis – Customer Segmentation Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/) 
[![Skills](https://img.shields.io/badge/Skills-RFM_Analysis-green)](https://www.linkedin.com/in/yourprofile)
[![Machine Learning](https://img.shields.io/badge/ML-KMeans_Clustering-orange)](https://github.com/yourprofile)

A comprehensive Python-based retail analytics project for customer segmentation using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering on **1,000 transactions from 100 customers** spanning **24 months** to identify high-value customers and optimize marketing strategies.

---

## 📋 Table of Contents

- [Project Objective](#project-objective)
- [Dataset Description](#dataset-description)
- [Key Analysis Steps](#key-analysis-steps)
- [Installation & Prerequisites](#installation--prerequisites)
- [How to Use](#how-to-use)
- [Key Findings](#key-findings)
- [Visualization Guide](#visualization-guide)
- [Learning Outcomes](#learning-outcomes)
- [Tech Stack](#tech-stack)

---

## 🎯 Project Objective

Segment retail customers based on purchasing patterns to:

✅ Identify high-value (VIP) customers  
✅ Identify inactive (churned) customers  
✅ Profile recent buyers and medium-value customers  
✅ Develop targeted marketing strategies per segment  
✅ Optimize customer lifetime value and retention  

---

## 📊 Dataset Description

### File: `synthetic_retail_transactions_1000.csv`

**Dataset Statistics:**
- **Total Records:** 1,000 transactions
- **Unique Customers:** 100 (C001-C100)
- **Data Period:** 24 months
- **Invoice Range:** 1001-2000
- **Transaction Amount Range:** Variable
- **Clean Data Quality:** No missing values after cleaning

### Column Definitions

| Column | Data Type | Description |
|--------|-----------|-------------|
| CustomerID | String | Unique customer identifier (C001-C100) |
| InvoiceNo | Integer/String | Unique transaction invoice number |
| InvoiceDate | DateTime | Date of transaction (YYYY-MM-DD format) |
| Amount | Float | Transaction amount in USD |

### Data Quality

**Before Cleaning:**
- Total records: 1,000 transactions
- 100 unique customers
- All records valid after cleaning

**After Cleaning:**
- Removed null CustomerIDs (if any)
- Removed zero/negative amounts (if any)
- All 1,000 records retained
- Data types standardized (InvoiceNo as string)

---

## 🔬 Key Analysis Steps

### Step 1: Load & Clean Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("synthetic_retail_transactions_1000.csv", parse_dates=['InvoiceDate'])
df.head()

# Data cleaning
df = df.dropna(subset=['CustomerID'])
df = df[df['Amount'] > 0]
df['InvoiceNo'] = df['InvoiceNo'].astype(str)

print(df.info())
```

### Step 2: Calculate RFM Metrics

```python
# Define snapshot date (1 day after last transaction)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Aggregate RFM metrics per customer
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',                                     # Frequency
    'Amount': 'sum'                                           # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print(rfm.shape)
```

### Step 3: Explore RFM Data

```python
# Summary statistics
print(rfm.describe())

# Visualize distributions
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
sns.histplot(rfm['Recency'], kde=True, bins=20, color="skyblue")
plt.title('Recency Distribution')

plt.subplot(1,3,2)
sns.histplot(rfm['Frequency'], kde=True, bins=20, color="lightgreen")
plt.title('Frequency Distribution')

plt.subplot(1,3,3)
sns.histplot(rfm['Monetary'], kde=True, bins=20, color="salmon")
plt.title('Monetary Distribution')

plt.tight_layout()
plt.show()
```

### Step 4: Scale RFM Data

```python
# Standardize RFM metrics for clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

print(rfm_scaled[:5])
```

### Step 5: Elbow Method for Optimal K

```python
# Test K values from 2 to 9
inertia = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.title("Elbow Method for Optimal k")
plt.show()
```

### Step 6: Apply K-Means Clustering

```python
# Fit K-Means with K=4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Cluster summary
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(1)

print(cluster_summary)
```

### Step 7: Visualize Customer Segments

```python
# Create scatter plots
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Frequency vs Monetary
sns.scatterplot(
    data=rfm, x="Frequency", y="Monetary",
    hue="Cluster", palette="Set2", s=100, ax=axes[0], edgecolor="k"
)
axes[0].set_title("Customer Segments (Frequency vs Monetary)", fontsize=14)

# Plot 2: Recency vs Frequency
sns.scatterplot(
    data=rfm, x="Recency", y="Frequency",
    hue="Cluster", palette="Set2", s=100, ax=axes[1], edgecolor="k"
)
axes[1].set_title("Customer Segments (Recency vs Frequency)", fontsize=14)

plt.tight_layout()
plt.show()
```

### Step 8: Create Cluster Profile Heatmap

```python
# Calculate mean RFM values per cluster
cluster_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cluster_profile, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Average RFM Values per Cluster", fontsize=14)
plt.show()

print(cluster_profile)
```

### Step 9: Assign Segment Labels

```python
# Map clusters to business-meaningful labels
cluster_labels = {
    0: "Churned / Inactive", 
    1: "Medium-Value Active", 
    2: "Recent Buyers", 
    3: "VIP / High-Value"
}

rfm['Segment'] = rfm['Cluster'].map(cluster_labels)

# Segment distribution
segment_counts = rfm['Segment'].value_counts()
print(segment_counts)

# Visualize distribution
plt.figure(figsize=(6,4))
sns.countplot(
    x='Segment',
    data=rfm,
    order=segment_counts.index,
    color="skyblue"
)
plt.title("Customer Segments Distribution", fontsize=14)
plt.xlabel("Segment", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.xticks(rotation=30)
plt.show()
```

---

## ⚙️ Installation & Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Python Version:** 3.8+

---

## 📈 Key Findings

### RFM Metrics Summary

| Metric | Min | Max | Mean | Std Dev |
|--------|-----|-----|------|---------|
| **Recency (days)** | 1 | 464 | 72.24 | 78.70 |
| **Frequency (purchases)** | 4 | 18 | 10.00 | 2.92 |
| **Monetary (USD)** | $3,294.56 | $18,622.56 | $10,307.94 | $3,382.78 |

### Cluster Breakdown

| Cluster | Label | Count | Avg Recency | Avg Frequency | Avg Monetary |
|---------|-------|-------|-------------|---------------|--------------|
| **0** | Churned / Inactive | 7 | 301.9 days | 7.6 | $7,951.40 |
| **1** | Medium-Value Active | 42 | 60.3 days | 10.0 | $10,407.30 |
| **2** | Recent Buyers | 28 | 46.1 days | 7.2 | $6,885.70 |
| **3** | VIP / High-Value | 23 | 55.9 days | 14.1 | $15,009.90 |

### Segment Distribution

- **Churned / Inactive:** 7 customers (7%)
- **Medium-Value Active:** 42 customers (42%)
- **Recent Buyers:** 28 customers (28%)
- **VIP / High-Value:** 23 customers (23%)

### Segment Characteristics

**Churned / Inactive (7 customers)**
- Highest recency: 301.9 days since last purchase
- Lowest frequency: 7.6 transactions
- Lowest monetary: $7,951.40

**Medium-Value Active (42 customers)**
- Moderate recency: 60.3 days
- Good frequency: 10.0 transactions
- Solid monetary value: $10,407.30
- Largest segment: 42% of customer base

**Recent Buyers (28 customers)**
- Low recency: 46.1 days
- Lower frequency: 7.2 transactions
- Lower monetary: $6,885.70

**VIP / High-Value (23 customers)**
- Recent engagement: 55.9 days
- Highest frequency: 14.1 transactions
- Highest monetary value: $15,009.90

---

## 📊 Visualization Guide

| Visualization | Description |
|---------------|-------------|
| **RFM Histograms** | Shows distribution of Recency, Frequency, and Monetary metrics |
| **Elbow Curve** | Displays inertia for K=2 to K=9, identifies optimal K |
| **Frequency vs Monetary Scatter** | Shows spending patterns and customer value by cluster |
| **Recency vs Frequency Scatter** | Shows engagement patterns and loyalty by cluster |
| **RFM Heatmap** | Compares average RFM values across clusters |
| **Segment Distribution** | Shows customer count per segment |

---

## 🎓 Learning Outcomes

✅ RFM analysis framework and customer segmentation principles

✅ Feature engineering and data standardization for clustering

✅ K-Means clustering algorithm and hyperparameter tuning

✅ Elbow method for determining optimal cluster count

✅ Data visualization with Matplotlib and Seaborn

✅ Customer behavior analysis and profiling

✅ Pandas data manipulation and aggregation

✅ Scikit-learn preprocessing and machine learning

✅ Business interpretation of clustering results

✅ Actionable insights from customer segmentation

---

## 📝 Notes

- Dataset contains 1,000 clean transactions from 100 unique customers
- RFM metrics calculated with snapshot date 1 day after latest transaction
- K-Means clustering applied with K=4 (determined by elbow method)
- All features standardized using StandardScaler before clustering
- Customer segments mapped to business-meaningful labels
- No missing values after data cleaning

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn |
| **Environment** | Jupyter Notebook / VS Code |
| **Dataset Used** | synthetic_retail_transactions_1000.csv |

## 📝 Author
**Robin Jimmichan Pooppally**  
[LinkedIn](https://www.linkedin.com/in/robin-jimmichan-pooppally-676061291) | [GitHub](https://github.com/Robi8995)

---

*This project demonstrates practical data science expertise in customer analytics, combining RFM analysis with machine learning to drive measurable improvements in customer segmentation, targeted marketing effectiveness, retention rates, and overall business revenue through data-driven customer intelligence*
