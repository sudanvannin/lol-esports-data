"""
Test script to check if DuckDB can successfully connect to the remote MotherDuck database
and query tables that are hosted there instead of reading local files.
"""

import os
import sys

import duckdb

def test_connection():
    # 1. Obter o Token do ambiente (igual o `db.py` fará)
    token = os.environ.get("MOTHERDUCK_TOKEN")
    
    if not token:
        print("❌ ERRO: A variável MOTHERDUCK_TOKEN não foi encontrada.")
        print("Crie sua conta em https://app.motherduck.com/, pegue o Token nas Settings e rode:")
        if sys.platform == "win32":
            print("  $env:MOTHERDUCK_TOKEN='seu_token_aqui'")
            print("  python test_motherduck.py")
        else:
            print("  export MOTHERDUCK_TOKEN='seu_token_aqui'")
            print("  python test_motherduck.py")
        return

    # 2. Conectar remotamente
    print(f"✅ Token encontrado! Conectando ao MotherDuck...")
    try:
        # A mágica acontece aqui: ao invés de ":memory:", conectamos via rede (md:lolesports)
        con = duckdb.connect(f"md:lolesports?motherduck_token={token}")
        
        # 3. Testar leitura
        print(f"✅ Conexão estabelecida com sucesso!")
        print("Vejamos as tabelas que existem lá na Nuvem:")
        
        tables = con.execute("SHOW TABLES").fetchall()
        for t in tables:
            print(f"  - Tabela remota: {t[0]}")
            
        print("\nPronto! A Web App usa exatamente essa lógica de conexão para achar os dados.")
        
    except Exception as e:
        print(f"❌ Falha ao conectar no MotherDuck: {e}")
        print("O banco de dados 'lolesports' já foi criado? (O script upload_to_motherduck.py faz isso)")

if __name__ == "__main__":
    test_connection()
