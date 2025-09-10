# ETL Plan (Draft)

- Imputation: median for numeric; mode or explicit NA level for categorical.
- Encoding: one-hot for categoricals (with rare-level binning).
- Scaling: as required by model family.
- Feature pruning: drop constant/near-constant and highly correlated numeric pairs.
