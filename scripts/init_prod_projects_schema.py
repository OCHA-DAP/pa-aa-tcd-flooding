"""One-off: create the `projects` schema in the PROD Postgres.

The dev DB (where projects.* monitoring tables lived) lost public network
access on 2026-09-22; prod had no `projects` schema at all. Run this ONCE via
a Databricks job (the Job Compute policy injects the prod write creds) before
the monitoring pipelines' first prod run; their pandas to_sql(if_exists=
"append") then creates the per-pipeline tables. Idempotent.

    python scripts/init_prod_projects_schema.py --mode prod
"""

import argparse

import ocha_stratus as stratus
from sqlalchemy import text

SQL = """
CREATE SCHEMA IF NOT EXISTS projects;
ALTER SCHEMA projects OWNER TO dbwriter;
GRANT USAGE, CREATE ON SCHEMA projects TO dbwriter;
GRANT USAGE ON SCHEMA projects TO dbreader;
ALTER DEFAULT PRIVILEGES FOR ROLE dbwriter IN SCHEMA projects
    GRANT SELECT ON TABLES TO dbreader;
ALTER DEFAULT PRIVILEGES IN SCHEMA projects
    GRANT SELECT ON TABLES TO dbreader;
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "prod"], required=True)
    args = ap.parse_args()
    engine = stratus.get_engine(args.mode, write=True)
    with engine.begin() as conn:
        for stmt in [x.strip() for x in SQL.split(";") if x.strip()]:
            print(stmt.splitlines()[0])
            conn.execute(text(stmt))
    print(f"projects schema ready in {args.mode}")
