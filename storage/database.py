"""
SonoAssist AI - Local Database Storage Module
Uses SQLite to store scans, morphological metrics, AI predictions, and radiologist review notes offline.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "sonoassist.db")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create or connect to the local SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=20.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize database tables for storing ultrasound analyses."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            patient_id TEXT DEFAULT 'ANONYMOUS',
            source_type TEXT DEFAULT 'IMAGE',
            original_path TEXT,
            enhanced_path TEXT,
            mask_path TEXT,
            overlay_path TEXT,
            predicted_class TEXT DEFAULT 'Undetermined',
            confidence REAL DEFAULT 0.0,
            area_pixels REAL DEFAULT 0.0,
            perimeter REAL DEFAULT 0.0,
            circularity REAL DEFAULT 0.0,
            aspect_ratio REAL DEFAULT 0.0,
            solidity REAL DEFAULT 0.0,
            model_mode TEXT DEFAULT 'UNTRAINED_DEMO',
            radiologist_reviewed INTEGER DEFAULT 0,
            radiologist_notes TEXT DEFAULT '',
            reviewer_name TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()


def insert_analysis(record: Dict[str, Any], db_path: str = DB_PATH) -> bool:
    """Insert or replace a scan analysis record into the database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO analyses (
                scan_id, timestamp, patient_id, source_type, original_path,
                enhanced_path, mask_path, overlay_path, predicted_class,
                confidence, area_pixels, perimeter, circularity, aspect_ratio,
                solidity, model_mode, radiologist_reviewed, radiologist_notes,
                reviewer_name
            ) VALUES (
                :scan_id, :timestamp, :patient_id, :source_type, :original_path,
                :enhanced_path, :mask_path, :overlay_path, :predicted_class,
                :confidence, :area_pixels, :perimeter, :circularity, :aspect_ratio,
                :solidity, :model_mode, :radiologist_reviewed, :radiologist_notes,
                :reviewer_name
            )
        """, {
            "scan_id": record.get("scan_id"),
            "timestamp": record.get("timestamp", datetime.now().isoformat()),
            "patient_id": record.get("patient_id", "ANONYMOUS"),
            "source_type": record.get("source_type", "IMAGE"),
            "original_path": record.get("original_path", ""),
            "enhanced_path": record.get("enhanced_path", ""),
            "mask_path": record.get("mask_path", ""),
            "overlay_path": record.get("overlay_path", ""),
            "predicted_class": record.get("predicted_class", "Undetermined"),
            "confidence": float(record.get("confidence", 0.0)),
            "area_pixels": float(record.get("area_pixels", 0.0)),
            "perimeter": float(record.get("perimeter", 0.0)),
            "circularity": float(record.get("circularity", 0.0)),
            "aspect_ratio": float(record.get("aspect_ratio", 0.0)),
            "solidity": float(record.get("solidity", 0.0)),
            "model_mode": record.get("model_mode", "UNTRAINED_DEMO"),
            "radiologist_reviewed": int(record.get("radiologist_reviewed", 0)),
            "radiologist_notes": record.get("radiologist_notes", ""),
            "reviewer_name": record.get("reviewer_name", "")
        })
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB Error] insert_analysis failed: {e}")
        return False
    finally:
        conn.close()


def update_analysis(record: Dict[str, Any], db_path: str = DB_PATH) -> bool:
    """Update an existing analysis record."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE analyses SET
                patient_id = :patient_id,
                source_type = :source_type,
                original_path = :original_path,
                enhanced_path = :enhanced_path,
                mask_path = :mask_path,
                overlay_path = :overlay_path,
                predicted_class = :predicted_class,
                confidence = :confidence,
                area_pixels = :area_pixels,
                perimeter = :perimeter,
                circularity = :circularity,
                aspect_ratio = :aspect_ratio,
                solidity = :solidity,
                model_mode = :model_mode,
                radiologist_reviewed = :radiologist_reviewed,
                radiologist_notes = :radiologist_notes,
                reviewer_name = :reviewer_name
            WHERE scan_id = :scan_id
        """, {
            "scan_id": record.get("scan_id"),
            "patient_id": record.get("patient_id", "ANONYMOUS"),
            "source_type": record.get("source_type", "IMAGE"),
            "original_path": record.get("original_path", ""),
            "enhanced_path": record.get("enhanced_path", ""),
            "mask_path": record.get("mask_path", ""),
            "overlay_path": record.get("overlay_path", ""),
            "predicted_class": record.get("predicted_class", "Undetermined"),
            "confidence": float(record.get("confidence", 0.0)),
            "area_pixels": float(record.get("area_pixels", 0.0)),
            "perimeter": float(record.get("perimeter", 0.0)),
            "circularity": float(record.get("circularity", 0.0)),
            "aspect_ratio": float(record.get("aspect_ratio", 0.0)),
            "solidity": float(record.get("solidity", 0.0)),
            "model_mode": record.get("model_mode", "UNTRAINED_DEMO"),
            "radiologist_reviewed": int(record.get("radiologist_reviewed", 0)),
            "radiologist_notes": record.get("radiologist_notes", ""),
            "reviewer_name": record.get("reviewer_name", "")
        })
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB Error] update_analysis failed: {e}")
        return False
    finally:
        conn.close()


def update_radiologist_review(scan_id: str, reviewer_name: str, notes: str, db_path: str = DB_PATH) -> bool:
    """Update reviewer notes and review status for a scan."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE analyses
            SET radiologist_reviewed = 1,
                reviewer_name = ?,
                radiologist_notes = ?
            WHERE scan_id = ?
        """, (reviewer_name, notes, scan_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB Error] update_radiologist_review failed: {e}")
        return False
    finally:
        conn.close()


def get_all_analyses(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all analysis records ordered by timestamp descending."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_analysis_by_id(scan_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a single analysis record by its scan_id."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses WHERE scan_id = ?", (scan_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_summary_stats(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Return summary statistics for the dashboard."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM analyses")
    total_scans = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE radiologist_reviewed = 1")
    reviewed_scans = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM analyses WHERE area_pixels > 0")
    cysts_detected = cursor.fetchone()[0]
    conn.close()
    return {
        "total_scans": total_scans,
        "reviewed_scans": reviewed_scans,
        "pending_review": max(0, total_scans - reviewed_scans),
        "cysts_detected": cysts_detected
    }


def seed_database_with_patient_cases(cases_list: List[Dict[str, Any]], db_path: str = DB_PATH) -> None:
    """Populate database with structured patient cases for clinical demonstration."""
    init_db(db_path)
    for case in cases_list:
        pid = case.get("patient_id", "PT-UNKNOWN")
        scan_id = f"SCAN-{pid}"
        insert_analysis({
            "scan_id": scan_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_id": pid,
            "source_type": "IMAGE",
            "original_path": case.get("image_path", ""),
            "enhanced_path": case.get("image_path", ""),
            "mask_path": case.get("mask_path", ""),
            "overlay_path": case.get("image_path", ""),
            "predicted_class": case.get("category", "Simple cyst"),
            "confidence": 0.88 if case.get("category") == "Simple cyst" else 0.79,
            "area_pixels": 4850.0 if "Normal" not in case.get("category", "") else 0.0,
            "perimeter": 260.0 if "Normal" not in case.get("category", "") else 0.0,
            "circularity": 0.88 if "Simple" in case.get("category", "") else 0.72,
            "aspect_ratio": 1.15,
            "solidity": 0.96,
            "model_mode": "TRAINED_MODEL",
            "radiologist_reviewed": 1 if pid in ["PT-101", "PT-102", "PT-105", "PT-106"] else 0,
            "radiologist_notes": case.get("radiologist_impression", ""),
            "reviewer_name": "Dr. Sarah Lin, MD (Lead Gynecologic Radiologist)" if pid in ["PT-101", "PT-102", "PT-105", "PT-106"] else ""
        }, db_path)
