# Day 1 — EDA and Dataset Exploration

## The Project
Building an intent classification system for NeuralForge AI.
Dataset: CLINC150 — 150 intents, 15250 train rows, 
3100 val, 5500 test.

## Key Functions

| Function | What It Does |
|---|---|
| pd.read_csv() | Load CSV into dataframe |
| df.shape | Returns rows and columns |
| df.head() | Shows first 5 rows |
| df.isnull().sum() | Counts missing values per column |
| df['col'].value_counts() | Frequency of each value |
| df['col'].nunique() | Number of unique values |
| np.mean() | Calculate mean of array |

## Key Concepts

### Intent
What the user is trying to do with their message.
Text is what they said. Intent is what they meant.

### Class Imbalance
When some classes have far more samples than others.
Model learns majority classes well and ignores minority classes.
Accuracy becomes misleading as a metric.

### Out of Scope
Queries that don't belong to any known intent.
Model must learn to say "I don't know" instead of 
forcing a wrong classification.

### Vectorized Operations
Applying math to entire array at once instead of looping.
NumPy does this in C under the hood making it 
50x faster than plain Python loops.

## Dataset Observations
- 150 intents + 1 out of scope class = 151 total
- Out of scope has 250 samples vs 100 for all others
- Some intents are dangerously similar
  e.g. schedule_meeting vs meeting_schedule
- Dataset spans 10 domains making classification harder
