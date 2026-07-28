import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# Initializing environment and DB engine
load_dotenv()
cost_db = os.getenv("COST_DB")
vector_db = os.getenv("VECTOR_DB")
engine = create_engine(
    url=vector_db,
    pool_pre_ping=True
)
df_cost_db = pd.read_csv(cost_db)

# Connecting to DB and pushing data
with engine.begin() as conn:
    df_cost_db.to_sql(
        name="cost_database",
        if_exists='replace',
        con=conn,
        schema="public"
    )

print(f"Data push complete, cost database setup complete, loaded {len(df_cost_db)} records.")