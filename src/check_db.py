import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

try:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = [t[0] for t in cur.fetchall()]
    
    print("\n=== NEON POSTGRESQL GROUND TRUTH ===")
    print(f"Total Tables: {len(tables)}")
    print("-------------------------------------")
    
    for t in tables:
        cur.execute(f'SELECT count(*) FROM "{t}";')
        count = cur.fetchone()[0]
        print(f"Table '{t}': {count} rows")
        
    cur.close()
    conn.close()
except Exception as e:
    print("DB Connection/Query Error:", str(e))