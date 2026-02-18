import psycopg2
from pgvector.psycopg2 import register_vector
from .config import settings

def get_conn():
    conn = psycopg2.connect(
        dbname=settings.pg_db,
        user=settings.pg_user,
        password=settings.pg_password,
        host=settings.pg_host,
        port=settings.pg_port,
    )
    register_vector(conn)
    return conn
