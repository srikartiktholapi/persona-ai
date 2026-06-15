import os
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = os.getenv(
    "PERSONA_DB_URL",
    "postgresql://read_only_mirror:QP_Clan_RDS%232026@clan-database.c1emavmzc6zh.ap-south-1.rds.amazonaws.com:5432/clan_mirror",
)


def _connect():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)


def fetch_personas():
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name FROM public.client_persona WHERE status = 1 ORDER BY name"
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_activities(persona_id: int):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, activity_name, description FROM public.client_persona_activity "
                "WHERE persona_id = %s AND status = 1 ORDER BY activity_name",
                (persona_id,),
            )
            return [dict(row) for row in cur.fetchall()]


def fetch_tasks(activity_id: int):
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, task_name, description FROM public.client_persona_activity_task "
                "WHERE persona_activity_id = %s AND status = 1 ORDER BY task_name",
                (activity_id,),
            )
            return [dict(row) for row in cur.fetchall()]
