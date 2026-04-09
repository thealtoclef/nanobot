from nanobot.db import Database


class FactStore:
    def __init__(self, db: Database, session_key: str):
        self._db = db
        self._session_key = session_key

    def add(self, content: str, category: str) -> int:
        return self._db.add_fact(self._session_key, content, category)

    def add_many(self, facts: list[tuple[str, str]]) -> None:
        self._db.add_facts(self._session_key, facts)

    def get_digest(self, max_tokens: int = 500) -> str:
        return self._db.get_fact_digest(self._session_key, max_tokens)

    def get_existing_facts_text(self) -> str:
        facts = self._db.get_facts(self._session_key)
        if not facts:
            return ""
        return "\n".join(f"- [{f.category}] {f.content}" for f in facts)
