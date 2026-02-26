from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from utils.config_reader import ConfigReader

cfg = ConfigReader()
cass = cfg.get("cassandra_props") or cfg.get("cassandra") or {}

cassandra_host = cass.get("host")
username = cass.get("username")
password = cass.get("password")
keyspace = cass.get("keyspace")

if not cassandra_host or not keyspace:
    raise RuntimeError("Missing Cassandra host/keyspace (config/env).")

auth_provider = PlainTextAuthProvider(username=username, password=password) if username and password else None
cluster = Cluster([cassandra_host], auth_provider=auth_provider)
session = cluster.connect(keyspace)

# Step 1: List all tables
session.set_keyspace(keyspace)
rows = session.execute(f"SELECT table_name FROM system_schema.tables WHERE keyspace_name='{keyspace}';")

print("\n📋 Available Tables:")
tables = [row.table_name for row in rows]
for idx, table in enumerate(tables, 1):
    print(f"{idx}. {table}")

# Step 2: Choose a table (manually or hardcode)
table_name = input("\nEnter the table name to inspect: ").strip()

# Step 3: Get table structure
columns = session.execute(f"""
    SELECT column_name, kind, type
    FROM system_schema.columns
    WHERE keyspace_name='{keyspace}' AND table_name='{table_name}';
""")

print(f"\n🔍 Structure of table '{table_name}':")
for column in columns:
    print(f"{column.column_name} ({column.kind}) - {column.type}")

# Close connection
cluster.shutdown()
