import sqlite3
import os
import glob
from PIL import Image

def audit_database():
    db_path = "sanskrypt.db"
    print("=== DATABASE AUDIT ===")
    if not os.path.exists(db_path):
        print("sanskrypt.db does not exist on filesystem.")
        return

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables found: {tables}")
    for (tname,) in tables:
        cursor.execute(f"SELECT COUNT(*) FROM `{tname}`")
        count = cursor.fetchone()[0]
        print(f"Table '{tname}': {count} total rows")
        if tname == "manuscripts":
            cursor.execute("SELECT COUNT(*) FROM manuscripts WHERE ocr_text IS NOT NULL AND length(ocr_text) > 0")
            print(f"  With OCR text: {cursor.fetchone()[0]}")
            cursor.execute("SELECT COUNT(*) FROM manuscripts WHERE translated_text IS NOT NULL AND length(translated_text) > 0")
            print(f"  With translation text: {cursor.fetchone()[0]}")
            cursor.execute("SELECT COUNT(*) FROM manuscripts WHERE explanation IS NOT NULL AND length(explanation) > 0")
            print(f"  With explanation: {cursor.fetchone()[0]}")
            cursor.execute("SELECT COUNT(*) FROM manuscripts WHERE keywords IS NOT NULL AND length(keywords) > 0")
            print(f"  With keywords: {cursor.fetchone()[0]}")
            cursor.execute("SELECT COUNT(*) FROM manuscripts WHERE classification IS NOT NULL AND length(classification) > 0")
            print(f"  With classification: {cursor.fetchone()[0]}")
            cursor.execute("SELECT COUNT(*) FROM manuscripts WHERE ocr_confidence IS NOT NULL")
            print(f"  With confidence score: {cursor.fetchone()[0]}")
            cursor.execute("SELECT COUNT(*) FROM manuscripts WHERE status='failed'")
            print(f"  Failed records: {cursor.fetchone()[0]}")
            cursor.execute("SELECT status, COUNT(*) FROM manuscripts GROUP BY status")
            print(f"  Status breakdown: {cursor.fetchall()}")
            cursor.execute("SELECT classification, COUNT(*) FROM manuscripts GROUP BY classification")
            print(f"  Classification breakdown: {cursor.fetchall()}")
            cursor.execute("SELECT id, title, ocr_confidence, classification, status, original_image_path FROM manuscripts ORDER BY id DESC LIMIT 5")
            print("  Top 5 recent records:")
            for r in cursor.fetchall():
                print("   ", r)
    conn.close()

def audit_dataset():
    print("\n=== DATASET / MANUSCRIPT IMAGES AUDIT ===")
    upload_orig = os.path.join("uploads", "original")
    files = glob.glob(os.path.join(upload_orig, "*"))
    files = [f for f in files if not f.endswith(".gitkeep")]
    print(f"Total files in uploads/original: {len(files)}")
    
    unique_names = set()
    dimensions = []
    formats = set()
    
    for fpath in files:
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lower()
        formats.add(ext)
        # Extract base name without prefix hash
        base_suffix = fname.split("_", 1)[1] if "_" in fname else fname
        unique_names.add(base_suffix)
        try:
            with Image.open(fpath) as img:
                w, h = img.size
                dimensions.append((w, h, w * h, fname))
        except Exception as e:
            print(f"Could not open image {fname}: {e}")

    print(f"Unique base manuscript names: {len(unique_names)}")
    print(f"Base names list: {sorted(list(unique_names))}")
    print(f"Image formats present: {formats}")
    if dimensions:
        min_res = min(dimensions, key=lambda x: x[2])
        max_res = max(dimensions, key=lambda x: x[2])
        print(f"Minimum resolution: {min_res[0]}x{min_res[1]} ({min_res[2]} pixels) - {min_res[3]}")
        print(f"Maximum resolution: {max_res[0]}x{max_res[1]} ({max_res[2]} pixels) - {max_res[3]}")
        resolutions = [f"{d[0]}x{d[1]}" for d in dimensions]
        print(f"Resolution range: {min_res[0]}x{min_res[1]} to {max_res[0]}x{max_res[1]}")

    print("\n=== GROUND TRUTH / REFERENCE DATA AUDIT ===")
    # Check if there are any ground truth annotations, reference translations, or labeled CSVs
    annotation_files = glob.glob("**/*.csv", recursive=True) + glob.glob("**/*.json", recursive=True)
    print("CSV/JSON files across workspace:", annotation_files)

if __name__ == "__main__":
    audit_database()
    audit_dataset()
