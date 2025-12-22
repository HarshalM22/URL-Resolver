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
        id,
        website,
        websiteURL_ownership,
        website_confidence,
    ):
        # query = """
        # INSERT INTO hospitals
        # (id, website, websiteURL_ownership, website_confidence)
        # VALUES (%s, %s, %s, %s)
        # """
        # self.cursor.execute(
        #     query,
        #     (id, website, websiteURL_ownership, website_confidence)
        # )

        update_query = """
                UPDATE hospitals
                SET
                    website = %s,
                    websiteURL_ownership = %s,
                    website_confidence = %s
                WHERE id = %s
                """
        self.cursor.execute(update_query,  (website, websiteURL_ownership, website_confidence, id))
        self.conn.commit()
