# Tensile Analysis Professional API

**Tensile Analysis API** is a specialized microservice designed for processing mechanical test results 
(metals, polymers, composites). The service automatically calculates Young's Modulus, 
Yield Strength (R_p), Ultimate Tensile Strength (UTS), and generates scientific plots and Excel reports.


---

## 🚀 Key Features
* **Automatic Parsing**: Support for multi-experiment TXT files (Non-ZWICK formats) with automated splitting.
* **Advanced Calculations**:
    * **Young's Modulus ($E$)**: Configurable calculation window based on **Strain** or **Stress**.
    * **Proof Stress ($R_p$)**: Parallel offset method (e.g., $0.2\%$ or $0.1\%$ offset).
    * **UTS**: Peak stress detection with corresponding strain coordinates.
* **Scientific Visualization**: 
    * Individual plots for each specimen with dotted projection lines to axes.
    * **Combined Analysis**: A single plot overlaying all test curves for easy comparison.
* **Statistical Intelligence**: 
    * **Outlier Detection**: Automated marking of anomalous tests based on Z-score.
    * **Batch Statistics**: Calculation of **Mean** and **Standard Deviation** (excluding outliers).
* **Professional Reporting**: Multi-sheet Excel export (`.xlsx`) containing raw data and summary results.
* **Privacy-First**: Invisible session management via Cookies (UUID-based isolation).

---

## 🛠 Tech Stack
* **FastAPI** (Python 3.9+)
* **Pandas & NumPy** (Data processing)
* **Matplotlib** (Scientific plotting)
* **OpenPyXL** (Excel generation)
* **Docker & Docker Compose** (Containerization)

---

## 📦 Quick Start (Docker)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/tensile-analysis-api.git](https://github.com/your-username/tensile-analysis-api.git)
   cd tensile-analysis-api