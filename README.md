# F1 Driver Style Clustering - Thesis Project

A comprehensive data science pipeline for clustering Formula 1 drivers based on their driving styles using telemetry data and behavioral analysis.

## 🎯 Project Overview

This thesis project aims to **identify and cluster Formula 1 drivers based on their driving style** using aggregated telemetry and session data. The approach focuses on maintaining interpretability while analyzing driving behavior at granular levels (corner-specific, track-specific, and race-specific).

**Key Innovation**: The pipeline uses **spline-based distance alignment** to ensure consistent driver comparisons by mapping all telemetry data to a common reference frame, solving the fundamental challenge of inconsistent racing lines between drivers.

### Research Goals
- **Primary**: Cluster F1 drivers by telemetry-derived behavioral features
- **Secondary**: Maintain maximum granularity (corner/track/race level) for interpretability
- **Method**: Distance-based clustering with stratified approaches for enhanced interpretation
- **Validation**: Map clusters to real-world driving behavior, not just statistical artifacts

## 📊 Dataset Characteristics

### Scale & Structure
- **Observation Unit**: Each row represents a specific driver's approach to a specific corner type (fast/medium/slow) on a specific track for a given race
- **Total Observations**: ~3,828 observations (across 2022-2024 seasons)
- **Features**: 8 aggregated telemetry-derived features per observation
- **Drivers**: 28 unique drivers across multiple seasons
- **Tracks**: Complete F1 calendar coverage with corner-type classification

### Feature Set
Each observation aggregates telemetry into these behavioral indicators:
- **Speed Metrics**: Average entry/exit speeds, speed consistency
- **Acceleration Patterns**: Maximum acceleration into/out of corners
- **Braking Behavior**: Maximum deceleration patterns
- **Throttle Management**: Average throttle application into/out of corners
- **Control Metrics**: Speed consistency and control smoothness

### Data Granularity Strategy
The project maintains **corner-level, track-level, and session-level distinctions** to preserve meaningful differences in driving behavior rather than over-aggregating to season averages only.

## 📁 Project Structure

```
├── main.py                     # CLI interface for data extraction
├── two_stage_clustering.ipynb  # Interactive clustering analysis
│
├── Pipeline/                   # Data extraction and processing
│   ├── telemetry_extraction.py    # Core spline alignment system  
│   ├── corner_analysis.py         # Corner behavioral analysis
│   ├── session_analysis.py        # Race session processing
│   ├── track_analysis.py          # Circuit-specific analysis
│   ├── data_aggregation.py        # Multi-level aggregation
│   └── EXTRACTION_TECHNICAL_GUIDE.md  # Technical methodology
│
├── Clustering/                 # Machine learning pipeline
│   ├── workflow.py                 # Complete clustering workflow
│   ├── feature_engineering.py     # Feature matrix creation
│   ├── kmeans_clustering.py       # Clustering algorithms
│   ├── distance_metrics.py        # Distance calculations
│   ├── optimization.py            # Cluster optimization
│   └── analysis.py                # Results interpretation
│
├── cache/                      # FastF1 data cache (auto-generated)
├── corner_data/               # Processed corner analysis data
├── output/                    # Generated analysis results
└── data_processing/           # Additional processing utilities
```

## 🏗️ Architecture

### Core Modules

#### 1. Pipeline Module (`/Pipeline/`)
**Data Extraction & Processing Pipeline**
- `telemetry_extraction.py`: Core spline-based alignment system
- `corner_analysis.py`: Corner-specific behavioral analysis  
- `session_analysis.py`: Race session processing
- `track_analysis.py`: Circuit-specific analysis
- `data_aggregation.py`: Multi-level data aggregation
- `data_io.py`: Standardized I/O operations

**Key Innovation**: Uses periodic cubic splines and k-d tree spatial indexing to align all driver telemetry to a common reference path.

#### 2. Clustering Module (`/Clustering/`)
**Machine Learning & Analysis Pipeline**
- `workflow.py`: Complete clustering analysis pipeline
- `feature_engineering.py`: Driver feature matrix creation
- `kmeans_clustering.py`: K-means clustering implementation  
- `distance_metrics.py`: Custom distance calculations
- `optimization.py`: Cluster optimization (elbow method, silhouette analysis)
- `analysis.py`: Cluster interpretation and validation

### Current Implementation Status

**✅ Completed Components:**
- ✅ Data extraction pipeline with spline alignment
- ✅ Corner classification and behavioral analysis
- ✅ Feature engineering and standardization
- ✅ Two-stage clustering methodology
- ✅ Comprehensive data validation and quality checks
- ✅ Results interpretation and driver profiling

**🔬 Analysis Results (2022-2024 Data):**
- **Behavioral Clusters**: 3 distinct driving behavior patterns identified
- **Driver Style Groups**: 3 higher-level driver style archetypes
- **Coverage**: 28 drivers, ~23 races per season, all circuit types


## 🔧 Technical Methodology

### Spline-Based Alignment System
The core technical innovation addresses the fundamental challenge that **no two F1 laps are identical** - drivers take different racing lines, making raw distance measurements incomparable.

**Solution Process:**
1. **Reference Selection**: Use session's fastest lap as ideal racing line
2. **Spline Creation**: Fit periodic cubic spline to (X,Y) coordinates  
3. **Spatial Indexing**: Build k-d tree for efficient position mapping
4. **Data Alignment**: Map all telemetry to common reference distance
5. **Normalization**: Create comparable metrics across all drivers

This ensures **every data point from every driver is mapped to a common frame of reference**, enabling meaningful statistical comparisons.

### Feature Engineering Strategy
- **Standardization**: Z-score normalization for all features
- **Granularity Preservation**: Maintain corner/track/race distinctions
- **Behavioral Focus**: Features capture driving style, not car performance
- **Statistical Validity**: Aligned data enables proper cross-driver analysis

### Clustering Validation
- **Silhouette Analysis**: Optimal cluster count determination
- **Interpretability**: Map clusters to real-world driving behaviors
- **Cross-Validation**: Results consistent across multiple seasons
- **Domain Knowledge**: Validate against known driver characteristics

## 📈 Current Results Summary

### Two-Stage Clustering Approach

**Stage 1: Behavioral Clustering**
- **Method**: Agglomerative clustering on standardized telemetry features
- **Result**: 3 distinct behavioral clusters identified
  - **Cluster 0** (61.1%): Dominant general driving pattern
  - **Cluster 1** (29.9%): Alternative approach to cornering
  - **Cluster 2** (9.1%): Specialized/extreme driving behaviors

**Stage 2: Driver Style Profiling**  
- **Method**: K-means clustering on driver-specific cluster usage profiles
- **Result**: 3 driver style archetypes
  - **Style Group 0**: Conservative/consistent drivers (VET, MSC, LAT, DOO)
  - **Style Group 1**: Mainstream competitive group (VER, HAM, LEC, NOR, etc.)
  - **Style Group 2**: Outlier/unique approach (BEA)

### Key Findings
- **Vettel** shows highest usage of Cluster 0 behavior (70% of observations)
- **Piastri** demonstrates most frequent use of rare Cluster 2 behaviors (13% usage)
- **Most drivers** fall into the mainstream competitive group with similar cluster usage patterns
- **Clear separation** between conservative and aggressive cornering approaches

## 🚀 Quick Start

### Prerequisites
```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install fastf1 pandas numpy scipy scikit-learn tqdm matplotlib seaborn plotly openpyxl
```

### Basic Usage

#### 1. Data Extraction
```bash
# Extract season-wide driver features
python main.py --mode season --year 2024 --drivers VER,HAM,LEC

# Extract corner-specific telemetry for clustering
python main.py --mode corner --year 2024 --drivers VER,HAM,LEC

# Extract single race analysis
python main.py --mode single --year 2024 --race "Monaco" --drivers VER,HAM,LEC
```

#### 2. Clustering Analysis
```python
# Import clustering workflow
from Clustering.workflow import run_clustering_analysis

# Load corner data
corner_data = pd.read_csv("corner_data/corner_data2024.csv")

# Run complete clustering pipeline
results = run_clustering_analysis(
    corner_data, 
    granularity='corner_specific',
    distance_method='euclidean'
)
```

#### 3. Interactive Analysis
The `two_stage_clustering.ipynb` notebook provides a complete interactive analysis workflow with visualizations.


## 📚 Documentation

- **`README_MAIN_SCRIPT.md`**: Detailed CLI usage guide and examples
- **`Pipeline/EXTRACTION_TECHNICAL_GUIDE.md`**: Technical implementation details
- **`two_stage_clustering.ipynb`**: Interactive analysis with full methodology


## 📄 License

Academic research project. Please cite appropriately if using methodologies or results.

---

**Project Status**: Active thesis research with comprehensive clustering pipeline implemented and validated on 2022-2024 F1 seasons.