# PostDoc-project-1.1.1.9-LZP-1-24-083

**Innovative machine learning QSPR-SMILES modeling toolkit for polymer and composite material discovery and life prediction under environmental aging conditions**

---

## Review
Objective: end-to-end pipeline from experiments → DB → external data search → DB → model training → prediction of durability of polymers and composites during aging.

```mermaid
flowchart LR
  A[Raw experiments (Excel)] --> B[scripts/tensile batch_tensile.py]
  B --> C[Out files (CSV, Excel, PNG, HTML)]
  B --> D[(Database)]
  E[External data (articles, repos)] --> F[Parsing / ingestion]
  F --> D
  D --> G[ML QSPR-SMILES training]
  G --> H[Predictions]
  H --> I[Backend API] --> J[Frontend UI]
