import os
import sqlite3
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# Hardcoded magic numbers and mixed responsibilities
def process_user_data(
    db_path: str,
    input_file: str,
    output_file: str,
    min_age: int,
    max_age: int,
    country_filter: str,
    log_to_db: bool,
):
    """
    Intentionally bad function:
    - Does validation
    - Does DB I/O
    - Does file I/O
    - Does data transformation
    - Logs in multiple places
    All in one place (mixed responsibilities).
    """
    # Validation logic
    if not os.path.exists(input_file):
        logger.error("Input file does not exist")
        raise FileNotFoundError(input_file)

    if min_age < 0 or max_age > 120:
        logger.warning("Suspicious age range, using defaults 18-99")
        min_age = 18
        max_age = 99

    # Database connection and table creation (I/O)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            country TEXT
        )
        """
    )
    conn.commit()

    # Read input and transform (file I/O + transformation)
    users: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                # Ignore bad lines
                continue
            name, age_str, country = parts
            try:
                age = int(age_str)
            except ValueError:
                continue

            if age < min_age or age > max_age:
                continue

            if country_filter and country != country_filter:
                continue

            users.append({"name": name, "age": age, "country": country})

    # Write filtered users to DB (more I/O)
    for user in users:
        cur.execute(
            "INSERT INTO users (name, age, country) VALUES (?, ?, ?)",
            (user["name"], user["age"], user["country"]),
        )
    conn.commit()

    # Optional logging into DB (yet another concern)
    if log_to_db:
        cur.execute(
            "INSERT INTO users (name, age, country) VALUES (?, ?, ?)",
            ("LOG_ENTRY", 0, "SYSTEM"),
        )
        conn.commit()

    # Write a simple report file (file I/O again)
    total = len(users)
    avg_age = sum(u["age"] for u in users) / total if total > 0 else 0

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"Total users: {total}\n")
        out.write(f"Average age: {avg_age}\n")
        out.write(f"Country filter: {country_filter or 'ANY'}\n")

    logger.info("Processing complete")
    conn.close()


# Another long function with nesting and magic values
def transform_and_export(data):
    """
    Another intentionally bad function to test:
    - No type hints
    - Deep nesting
    - Magic values
    - Mixed responsibilities
    """
    results = []
    for item in data:
        if "status" in item:
            if item["status"] == "active":
                if "score" in item:
                    if item["score"] > 50:
                        if "country" in item:
                            if item["country"] == "NL":
                                # Deeply nested logic
                                item["priority"] = "high"
                            elif item["country"] == "BE":
                                item["priority"] = "medium"
                            else:
                                item["priority"] = "low"
                        else:
                            item["priority"] = "unknown"
                    else:
                        item["priority"] = "low"
                else:
                    item["priority"] = "unknown"
            else:
                item["priority"] = "inactive"
        else:
            item["priority"] = "missing"

        # Magic threshold values and side effects
        if item.get("score", 0) > 90:
            item["badge"] = "gold"
        elif item.get("score", 0) > 75:
            item["badge"] = "silver"
        elif item.get("score", 0) > 60:
            item["badge"] = "bronze"

        results.append(item)

    # Fake export (more responsibilities)
    with open("export.txt", "w", encoding="utf-8") as f:
        for row in results:
            f.write(str(row) + "\n")

    return results
