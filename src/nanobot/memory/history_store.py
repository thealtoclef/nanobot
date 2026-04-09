from nanobot.db import Database


class HistoryStore:
    def __init__(self, db: Database, session_key: str):
        self._db = db
        self._session_key = session_key

    def add(self, summary: str, summarized_through_message_id: int | None) -> int:
        return self._db.add_history(self._session_key, summary, summarized_through_message_id)

    def get_current_summary(self) -> str | None:
        return self._db.get_latest_history_summary(self._session_key)
