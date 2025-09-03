# PostDoc-project-1.1.1.9-LZP-1-24-083

**Innovative machine learning QSPR-SMILES modeling toolkit for polymer and composite material discovery and life prediction under environmental aging conditions**

---

## Overview
This project builds a full pipeline from experimental data to machine learning predictions of polymer/composite durability under environmental aging.

```mermaid
flowchart LR
  A[Raw experiments] --> B[Tensile scripts]
  B --> C[Reports & Plots]
  B --> D[(Database)]
  E[External datasets] --> F[Ingestion]
  F --> D
  D --> G[ML QSPR-SMILES]
  G --> H[Predictions]
  H --> I[Backend API]
  I --> J[Frontend UI]
