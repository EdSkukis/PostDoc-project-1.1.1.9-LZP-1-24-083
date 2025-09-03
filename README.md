# PostDoc-project-1.1.1.9-LZP-1-24-083

**Innovative machine learning QSPR-SMILES modeling toolkit for polymer and composite material discovery and life prediction under environmental aging conditions**

---

## Review
Objective: end-to-end pipeline from experiments → DB → external data search → DB → model training → prediction of durability of polymers and composites during aging.

```mermaid
flowchart LR
  A[Raw experiments<br>(Excel .xls/.xlsx)] --> B[scripts/tensile/batch_tensile.py<br>очистка, метрики, графики]
  B --> C[out/*<br>Combined.csv, Summary.csv, Report.xlsx, PNG, HTML]
  B --> D[(DB)]
  E[Внешние данные<br>статьи, репозитории] --> F[Парсинг/ингест]
  F --> D
  D --> G[ML/QSPR-SMILES обучение]
  G --> H[Предсказания/аналитика]
  H --> I[backend API] --> J[frontend UI]
