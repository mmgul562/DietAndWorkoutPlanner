import sqlite3
import csv
from pathlib import Path
from config import FOOD_DB_PATH


TSV_PATH = "sources/openfoodfacts.tsv"


def safe_float(val):
    try:
        return float(val) if val and val.strip() else None
    except:
        return None


def safe_int(val):
    try:
        return int(float(val)) if val and val.strip() else None
    except:
        return None


def import_tsv():
    # Ensure directory exists
    db_dir = Path(FOOD_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)


    csv.field_size_limit(10 * 1024 * 1024)

    # Delete old database if it exists (force clean)
    db_path = Path(FOOD_DB_PATH)
    if db_path.exists():
        print(f"Removing old database: {FOOD_DB_PATH}")
        db_path.unlink()

    # Connect to new database
    conn = sqlite3.connect(FOOD_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cursor = conn.cursor()

    # Create products table (no FTS yet)
    cursor.execute("""
                   CREATE TABLE products
                   (
                       code          TEXT PRIMARY KEY,
                       product_name  TEXT,
                       brands        TEXT,
                       categories    TEXT,
                       energy_kcal   REAL,
                       fat           REAL,
                       saturated_fat REAL,
                       carbs         REAL,
                       sugars        REAL,
                       proteins      REAL,
                       fiber         REAL,
                       salt          REAL,
                       nutriscore    TEXT,
                       nova_group    INTEGER
                   )
                   """)
    conn.commit()

    # Import TSV data
    if not Path(TSV_PATH).exists():
        print(f"TSV file not found: {TSV_PATH}")
        return

    batch = []
    batch_size = 10000
    total = 0
    inserted = 0

    with open(TSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            # Basic filtering
            if not row.get("product_name") or not safe_float(row.get("energy-kcal_100g")):
                continue
            try:
                code = row["code"].strip()
                name = row["product_name"][:200]
                brands = row.get("brands", "")[:100]
                categories = row.get("categories_tags", "")[:200]
                energy_kcal = float(row["energy-kcal_100g"])
                fat = safe_float(row.get("fat_100g"))
                sat_fat = safe_float(row.get("saturated-fat_100g"))
                carbs = safe_float(row.get("carbohydrates_100g"))
                sugars = safe_float(row.get("sugars_100g"))
                proteins = safe_float(row.get("proteins_100g"))
                fiber = safe_float(row.get("fiber_100g"))
                salt = safe_float(row.get("salt_100g"))
                nutriscore = row.get("nutriscore_grade", "").upper()
                nova = safe_int(row.get("nova_group"))

                batch.append((code, name, brands, categories, energy_kcal, fat, sat_fat,
                              carbs, sugars, proteins, fiber, salt, nutriscore, nova))

                if len(batch) >= batch_size:
                    cursor.executemany("""
                                       INSERT
                                       OR IGNORE INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                       """, batch)
                    conn.commit()
                    inserted += len(batch)
                    print(f"Inserted {inserted} products (processed {total} rows)")
                    batch.clear()
            except Exception:
                continue

    if batch:
        cursor.executemany("INSERT OR IGNORE INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        conn.commit()
        inserted += len(batch)

    print(f"Inserted {inserted} products into 'products' table.")

    # Now create FTS5 table (drop any existing)
    print("Creating full-text search table...")
    cursor.execute("DROP TABLE IF EXISTS products_fts")
    cursor.execute("""
        CREATE VIRTUAL TABLE products_fts USING fts5(
            code, product_name, brands, categories,
            content=products
        )
    """)

    # Populate FTS index
    print("Populating FTS index...")
    cursor.execute("""
                   INSERT INTO products_fts(rowid, code, product_name, brands, categories)
                   SELECT rowid, code, product_name, brands, categories
                   FROM products
                   """)
    conn.commit()

    # Verify
    count = cursor.execute("SELECT COUNT(*) FROM products_fts").fetchone()[0]
    print(f"FTS index built with {count} entries.")

    conn.close()
    print("Database ready.")


if __name__ == "__main__":
    import_tsv()