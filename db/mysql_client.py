import mysql.connector
from config.settings import DB_CONFIG


class MySQLClient:
    def __init__(self):
        self.conn = mysql.connector.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor(dictionary=True)

    def fetch_unprocessed_hospitals(self, limit=10):
        query = """
        SELECT id, name, state
        FROM hospitals
        WHERE website is null
        LIMIT %s
        """
        self.cursor.execute(query, (limit,))
        return self.cursor.fetchall()

    def save_result(
        self,
        hospital_id,
        domain,
        ownership,
        confidence,
        status,
        error_message=None
    ):
        query = """
        INSERT INTO hospital_domains
        (hospital_id, domain, ownership, confidence, status, error_message)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.cursor.execute(
            query,
            (hospital_id, domain, ownership, confidence, status, error_message)
        )

        update_query = """
        UPDATE hospitals SET processed = TRUE WHERE id = %s
        """
        self.cursor.execute(update_query, (hospital_id,))
        self.conn.commit()
