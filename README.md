# Customer Support AIOps

An end-to-end MLOps pipeline that automatically classifies incoming support tickets by priority and uses Claude to draft professional responses. Experiment tracking and model registry are handled by MLflow, with infrastructure deployed on Azure ML.

## How it works

1. **Classify** — a TF-IDF + Random Forest model predicts ticket priority: `urgent`, `normal`, or `low`.
2. **Respond** — the predicted priority is passed to Claude (`claude-sonnet-4-5`) which drafts a context-aware reply.
3. **Monitor** — every interaction is logged to MLflow and appended to `logs/interactions.json` for drift analysis.

## Project structure

```
customer-support-aiops/
├── ml/src/
│   ├── train.py        # Train the classifier and register it in Azure ML
│   └── monitor.py      # Log interactions to MLflow + JSON
├── genai/
│   └── respond.py      # Load model, classify tickets, draft responses via Claude
├── infrastructure/
│   └── main.bicep      # Azure ML workspace, storage, Key Vault, compute cluster
├── logs/
│   └── interactions.json   # Append-only interaction log
└── mlflow.db           # Local SQLite MLflow tracking store
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install anthropic mlflow scikit-learn pandas azure-ai-ml azure-identity
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

## Usage

**Train and register the model:**

```bash
python ml/src/train.py
```

This runs a `support-ticket-classifier` experiment in MLflow and registers the trained pipeline in Azure ML.

**Run the response pipeline:**

```bash
python genai/respond.py
```

Classifies each ticket, drafts a response with Claude, and logs the interaction.

**View MLflow experiments:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Infrastructure (Azure)

The Bicep template in `infrastructure/main.bicep` provisions:

| Resource | Name pattern |
|---|---|
| Storage Account | `mlopsproject{env}st` |
| Key Vault | `mlops-project-{env}-kv` |
| Application Insights | `mlops-project-{env}-ai` |
| ML Workspace | `mlops-project-{env}-ws` |
| Compute Cluster | `training-cluster` (0–4 nodes, `Standard_DS3_v2`) |

Deploy with:

```bash
az deployment group create \
  --resource-group mlops-project-rg \
  --template-file infrastructure/main.bicep \
  --parameters environment=dev
```

## MLflow experiments

| Experiment | Tracks |
|---|---|
| `support-ticket-classifier` | accuracy, F1, model artifact |
| `support-ticket-monitoring` | response length, prediction correctness |
