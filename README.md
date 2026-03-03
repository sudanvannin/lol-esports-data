# LoL Esports Data Lakehouse

Pipeline de dados para coleta, processamento e analise de partidas competitivas de League of Legends.

## Stack

- **Orquestracao:** Apache Airflow
- **Storage:** MinIO (S3-compatible) + Delta Lake
- **Processamento:** Apache Spark (PySpark)
- **ML Tracking:** MLflow
- **Notebooks:** Jupyter

## Arquitetura

```
LoL Esports API ──┐
Match Feed API ───┼──> Python Collectors ──> Airflow ──> Bronze (Raw)
Oracle's Elixir ──┘                                          │
                                                             ▼
                                                    Spark Processing
                                                             │
                                                             ▼
                                                    Silver (Clean)
                                                             │
                                                             ▼
                                                    Gold (Features)
                                                             │
                                                             ▼
                                                    ML Training ──> MLflow
```

## Quick Start

```bash
# Copiar variaveis de ambiente
cp .env.example .env

# Subir infraestrutura
docker-compose up -d

# Acessar servicos
# Airflow:  http://localhost:8081 (admin/admin)
# MinIO:    http://localhost:9001 (minio/minio123)
# Spark:    http://localhost:8080
# MLflow:   http://localhost:5000
# Jupyter:  http://localhost:8888
```

## Desenvolvimento

```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Rodar testes
pytest tests/ -v

# Formatar codigo
black src/ dags/ tests/
ruff check src/ dags/ tests/ --fix
```

## Estrutura

```
lol-esports-data/
├── dags/                 # Airflow DAGs
├── src/
│   ├── ingestion/        # Coletores de dados
│   ├── processing/       # Jobs Spark
│   └── ml/               # Treinamento de modelos
├── notebooks/            # Exploracao
├── tests/                # Testes
└── docker-compose.yml    # Infraestrutura
```

## CI/CD

- **Lint:** Black, Ruff, MyPy
- **Tests:** Pytest com cobertura
- **Validacao:** DAGs Airflow, Jobs Spark

Pull requests para `main` passam por validacao automatica.
