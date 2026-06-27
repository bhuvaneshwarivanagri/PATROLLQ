# 🚔 PatrolIQ — Smart Safety Analytics Platform

> **GUVI | HCL Capstone Project** — Unsupervised Machine Learning · Public Safety Analytics

PatrolIQ is an interactive Streamlit dashboard built for the **Chicago Police Department** that applies unsupervised machine learning to crime data. It enables data-driven patrol scheduling, hotspot detection, and crime pattern analysis.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Dataset](#-dataset)
- [Dashboard Pages](#-dashboard-pages)
- [Machine Learning Models](#-machine-learning-models)
- [Key Insights](#-key-insights)
- [Team](#-team)

---

## 🔍 Project Overview

PatrolIQ analyzes Chicago crime records to uncover spatial and temporal patterns using clustering algorithms. The goal is to assist law enforcement with:

- Identifying crime **hotspot zones** across the city
- Understanding **when** crimes are most likely to occur
- Comparing clustering algorithms for the best patrol deployment strategy
- Enabling interactive **filter-based crime exploration**

---

## ✨ Features

- **Animated police car UI** with dark navy theme tailored for law enforcement context
- **KPI dashboard** — arrest rate, domestic crime rate, severity scores, crime type counts
- **7-tab Exploratory Data Analysis (EDA)** covering crime types, time, season, geography, arrest behavior, heatmaps, and feature correlations
- **3 clustering algorithms** — K-Means, DBSCAN, Hierarchical — on geographic and temporal data
- **Side-by-side model comparison** with Silhouette Score, Davies-Bouldin Index, and Elbow curve
- **Interactive Crime Analyzer** with multi-filter support and CSV export
- Fully **cached computations** for fast re-renders

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| Dashboard | `streamlit` |
| Data Processing | `pandas`, `numpy` |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (KMeans, DBSCAN, AgglomerativeClustering, PCA, StandardScaler) |
| Map Rendering | Plotly Mapbox (`carto-darkmatter` style) |

---

## 📁 Project Structure

```
patroliq/
│
├── app.py                      # Main Streamlit dashboard application
├── patrollq_complete.ipynb     # Jupyter Notebook — full analysis, EDA, and ML experiments
├── patrol.csv                  # Chicago crime dataset (place in same folder)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📦 Requirements

Create a `requirements.txt` file in your project folder with the following:

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
scikit-learn>=1.3.0
```

---

## ⚙️ Installation

**1. Clone or download this repository.**

**2. Install all dependencies at once using requirements.txt:**

```bash
pip install -r requirements.txt
```

**Or install manually:**

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
```

**3. Place `patrol.csv` in the same folder as `app.py`.**

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

---

## 📂 Dataset

The app reads from **`patrol.csv`** — a Chicago crime dataset expected to contain the following columns:

| Column | Description |
|---|---|
| `Date` | Date and time of the crime |
| `Primary Type` | Crime category (e.g., THEFT, BATTERY) |
| `Latitude` / `Longitude` | Geographic coordinates |
| `District` | Police district number |
| `Arrest` | Whether an arrest was made (boolean) |
| `Domestic` | Whether the crime was domestic (boolean) |
| `Ward`, `Community Area`, `Beat` | Administrative zone fields |
| `Location Description` | Type of location (e.g., STREET, RESIDENCE) |

The app engineers these additional features at load time:

| Engineered Feature | Description |
|---|---|
| `Hour`, `Month`, `Year` | Extracted from `Date` |
| `Day_of_Week`, `Day_Num` | Day name and number (0=Mon) |
| `Is_Weekend` | True if Saturday or Sunday |
| `Season` | Winter / Spring / Summer / Fall |
| `Time_of_Day` | Late Night / Morning / Afternoon / Evening |
| `Crime_Severity_Score` | Manual 1–10 severity scale per crime type |
| `Crime_Type_Encoded` | Label-encoded crime type for ML |

---

## 📊 Dashboard Pages

### 🏠 Home
Overview of the entire dataset with:
- 5 KPI cards (unique crime types, arrest rate, domestic rate, avg severity, high-severity %)
- Crime summary: top 5 crimes, severity breakdown, quick facts (peak hour, busiest day, etc.)
- Top 15 crimes horizontal bar chart colored by severity
- Crime highlight cards (15 crime types with count, severity, arrest rate)
- City-wide crime map colored by severity score

### 📊 EDA — Exploratory Data Analysis
Seven tabs covering the full data profile:

1. **Crime Types** — bar + pie chart, adjustable top-N slider
2. **Temporal** — crimes by hour, day of week, and yearly trend
3. **Seasonal** — season pie chart and monthly bar chart
4. **Geographic** — density heatmap and top districts by crime count
5. **Arrest & Domestic** — overall arrest rate, arrest rate by crime type, domestic crime breakdown
6. **Heatmaps** — Hour × Day of Week heatmap, Crime Type × Time of Day heatmap
7. **Correlation** — Seaborn heatmap of numeric feature correlations

### 🗺️ Geographic Clustering
Clusters crime locations on a Mapbox map using your choice of:
- **K-Means** — user-defined K, displays cluster centers as stars
- **DBSCAN** — tunable epsilon and min_samples, noise points highlighted separately
- **Hierarchical** — Ward linkage with user-defined K

Each view displays Silhouette Score and Davies-Bouldin Index.

### ⏰ Temporal Clustering
K-Means clustering on time-based features (Hour, Day, Month, Weekend, Severity). Shows cluster scatter plots, hour-by-cluster line charts, and a cluster profiles table identifying each cluster's dominant time window.

### 📈 Model Comparison
Runs all three algorithms on the same data sample and presents:
- Comparison table with scores and best-use recommendations
- Silhouette score bar chart with a 0.5 target line
- K-Means Elbow + Silhouette dual-axis chart
- Automatic best algorithm recommendation

### 🔍 Crime Analyzer
Interactive filter tool:
- Filter by crime type, hour range, and season
- View filtered metrics, bar charts, and crime map
- Preview up to 200 raw records
- Download filtered data as CSV

---

## 🤖 Machine Learning Models

| Algorithm | Type | Use Case |
|---|---|---|
| **K-Means** | Partitioning | Define fixed patrol zones; fast and interpretable |
| **DBSCAN** | Density-based | Find natural crime hotspots; handles noise |
| **Agglomerative (Hierarchical)** | Hierarchical | Build zone hierarchy; no need to pre-define K |

**Evaluation Metrics:**
- **Silhouette Score** — measures cluster separation (target > 0.5)
- **Davies-Bouldin Index** — measures cluster compactness (lower is better)
- **Inertia** — within-cluster sum of squares (used in Elbow method for K selection)

---

## 💡 Key Insights

- **THEFT and BATTERY** account for over 40% of all crimes
- **Crime peaks** at midnight and noon; Fridays and Saturdays are busiest
- **Summer (July/August)** has the highest crime volume; Winter has the least
- **Early morning (4–6 AM)** is consistently the safest window
- **Homicide** is the highest severity crime at 10/10; assigned dedicated patrol priority
- Temporal clusters with **high severity + late-night hours** represent the highest-risk patrol windows

---

## 👩‍💻 Team

**GUVI | HCL Capstone Project**
Unsupervised Machine Learning · Public Safety Analytics

Built for the **Chicago Police Department** crime intelligence initiative.

---

*"We Serve and Protect" — Chicago Police Department, Est. 1837*
