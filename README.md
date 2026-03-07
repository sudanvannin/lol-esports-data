# LoL Esports Data Lakehouse (Serverless)

Nascido da vontade de criar uma alternativa superior, mais rápida e independente ao *gol.gg*, este projeto evoluiu para um pipeline de dados **100% gratuito e automatizado** de partidas competitivas de League of Legends.

Ele abandona as arquiteturas de dados tradicionais pesadas (como clusters sempre ligados e Bancos SQL pagos) em favor de uma stack **Cloud-Native Serverless** inteligente. Usando **GitHub Actions**, **MotherDuck** e **Render**, o sistema consolida as métricas históricas de ouro, torneios e WinRates diariamente a custo zero, com queries retornando em milissegundos.

## Stack Tecnológica

- **Orquestração & ETL:** GitHub Actions 
- **Data Engine:** DuckDB (Pandas / SQL local)
- **Data Lake Cloud:** MotherDuck (Managed DuckDB)
- **Web App:** FastAPI + Jinja HTML (Deploy no Render)

---

## Arquitetura 

```text
LoL Esports API ──┐
Match Feed API ───┼──> GitHub Actions Runner ──> Bronze (Local JSON/CSV)
Oracle's Elixir ──┘    (Cron Diário/Semanal)             │
                                                         ▼
                                                Data Processing
                                                  (DuckDB)
                                                         │
                                                         ▼
    Render Web App <────── md:lolesports ──────  Silver (Parquets Locais)
   (FastAPI / HTML)           (Query)                    │
      (Grátis)                                           ▼
                            MotherDuck  <------- script upload_to_motherduck.py
```

## Como funciona?
1. **Cron Jobs**: Todo dia (API) e domingo (Elixir) o GitHub acorda a branch `.github/workflows`, baixa os dados crús e os joga num disco temporário.
2. **Pipelines Python**: Em seguida o Action processa os Parquets limpos localmente usando Pandas/DuckDB e joga na nuvem (MotherDuck) com um comando mágico.
3. **Frontend**: O site não tem banco de dados local. Toda vez que um usuário acessa, a `web/db.py` joga uma query remota (`md:lolesports`) direto no MotherDuck, que por usar um banco embarcado distribuído retorna tabelas ricas em milisegundos usando apenas 40MB da memória do Web Server.

---

## Quick Start (Deploy do Zero ao Herói)

**Aviso:** O projeto não exige mais instalação do Docker de dezenas de containers localmente.

### 1. Preparar o GitHub (ETL Database)
1. Crie uma conta gratuita em [MotherDuck.com](https://app.motherduck.com/) e gere seu Token.
2. No seu repositório no GitHub, vá em **Settings > Secrets and variables > Actions** e adicione as variáveis:
   - `MOTHERDUCK_TOKEN`
   - `LOL_ESPORTS_API_KEY` (Sua chave da Riot Games)
3. Na aba *Actions*, as schedules `ingestion_daily.yml` e `oracles_weekly.yml` cuidarão da ingestão.

### 2. Preparar e Ligar a Web App
Recomendado usar o [Render.com](https://render.com/) (Free Tier).

1. Crie um Web Service no Render apontando para este repositório (`main`).
2. Adicione nas **Environment Variables** do Render o seu `MOTHERDUCK_TOKEN`.
3. O Render encontrará automaticamente o `Dockerfile` raíz e as portas corretas. Feito!

---

## Desenvolvimento Local (Testando o UI)

Quer só rodar o site do seu VS Code local para ver as tabelas e gráficos da Riot?

```bash
# 1. Defina o token do Motherduck no seu terminal (Linux/Mac)
export MOTHERDUCK_TOKEN="seu_token_aqui"

# (PowerShell no Windows)
$env:MOTHERDUCK_TOKEN="seu_token_aqui"

# 2. Instale os requerimentos Web apenas
pip install -r requirements.txt

# 3. Suba o site
uvicorn web.app:app --host 127.0.0.1 --port 8000
```
Acesse `http://localhost:8000`. Não é preciso ter os Parquets pesados armazenados no seu SSD! O DuckDB pegará tudo da Cloud.

---

## Estrutura principal
```
lol-esports-data/
├── .github/workflows/    # Pipelines ETL (GitHub Actions substituindo o Airflow)
├── src/                  # Ferramentas originais de ingestão em Python
├── scripts/              # Transformações (Bronze -> Silver) e script do MotherDuck
├── web/                  # Web App FastAPI completa e páginas HTML (Jinja2)
├── Dockerfile            # Configuração level de Hosting (Render)
└── requirements.txt      # Dependências puras
```
