import importlib.util
import os
import json
import io
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout

import numpy as np

from embedding_codec import encode_embedding


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_script_module(module_name, script_filename):
    script_path = os.path.join(REPO_ROOT, script_filename)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def temporary_pipeline_config(config_data):
    config_path = os.path.join(REPO_ROOT, "pipeline_config.json")
    original_exists = os.path.exists(config_path)
    original_content = None

    if original_exists:
        with open(config_path, "r", encoding="utf-8") as fh:
            original_content = fh.read()

    try:
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh, indent=2)
        yield
    finally:
        if original_exists:
            with open(config_path, "w", encoding="utf-8") as fh:
                fh.write(original_content)
        else:
            os.remove(config_path)


def load_cluster_module_with_config(module_name, config_data):
    output = io.StringIO()
    with temporary_pipeline_config(config_data), redirect_stdout(output):
        module = load_script_module(module_name, os.path.join("scripts", "2_face_clustering.py"))
    return module, output.getvalue()


class TestConfidenceGatedAssigner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script_module("cluster_module", os.path.join("scripts", "2_face_clustering.py"))

    def test_high_confidence_assign(self):
        assigner = self.mod.ConfidenceGatedAssigner(
            {"high_threshold": 0.7, "low_threshold": 0.5, "margin_threshold": 0.08}
        )
        identities = [
            self.mod.IdentityRecord(person_id=1, centroid=np.array([1.0, 0.0])),
            self.mod.IdentityRecord(person_id=2, centroid=np.array([0.0, 1.0])),
        ]
        emb = np.array([0.99, 0.01])

        result = assigner.assign(emb, identities)

        self.assertEqual(result.status, "ASSIGNED")
        self.assertEqual(result.person_id, 1)

    def test_ambiguous_goes_to_review(self):
        assigner = self.mod.ConfidenceGatedAssigner(
            {"high_threshold": 0.7, "low_threshold": 0.5, "margin_threshold": 0.08}
        )
        identities = [
            self.mod.IdentityRecord(person_id=1, centroid=np.array([1.0, 0.0])),
            self.mod.IdentityRecord(person_id=2, centroid=np.array([0.99, 0.1])),
        ]
        emb = np.array([1.0, 0.05])

        result = assigner.assign(emb, identities)

        self.assertEqual(result.status, "REVIEW")

    def test_low_score_new_identity(self):
        assigner = self.mod.ConfidenceGatedAssigner(
            {"high_threshold": 0.9, "low_threshold": 0.8, "margin_threshold": 0.1}
        )
        identities = [
            self.mod.IdentityRecord(person_id=1, centroid=np.array([1.0, 0.0])),
            self.mod.IdentityRecord(person_id=2, centroid=np.array([0.0, 1.0])),
        ]
        emb = np.array([-1.0, -1.0])

        result = assigner.assign(emb, identities)

        self.assertEqual(result.status, "NEW_IDENTITY")


class TestCentroidManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script_module("cluster_module_cm", os.path.join("scripts", "2_face_clustering.py"))

    def test_centroid_not_updated_on_review(self):
        manager = self.mod.CentroidManager(0.72, 0.08)
        identity = self.mod.IdentityRecord(person_id=1, centroid=np.array([1.0, 0.0]), n_confirmed=5)
        original = identity.centroid.copy()

        result = self.mod.AssignmentResult(
            status="REVIEW",
            person_id=None,
            best_similarity=0.9,
            margin=0.2,
            candidate_ids=[1, 2],
            candidate_scores=[0.9, 0.7],
        )
        manager.update(identity, np.array([0.0, 1.0]), result)

        self.assertTrue(np.allclose(identity.centroid, original))
        self.assertEqual(identity.n_confirmed, 5)


class TestAssignmentPipelineFixes(unittest.TestCase):
    def test_escalated_faces_excluded_from_loader(self):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE faces (
                id INTEGER PRIMARY KEY,
                encoding BLOB,
                person_id INTEGER,
                assignment_status TEXT
            )
            """
        )
        cursor.execute(
            "INSERT INTO faces (id, encoding, person_id, assignment_status) VALUES (?, ?, ?, ?)",
            (1, encode_embedding(np.array([1.0, 0.0], dtype=np.float32)), -1, "ESCALATED"),
        )

        cursor.execute(
            """
            SELECT id, encoding, assignment_status
            FROM faces
            WHERE (person_id IS NULL OR person_id = -1)
              AND (assignment_status IS NULL
                   OR (assignment_status != 'REVIEW'
                       AND assignment_status != 'ESCALATED'))
            """
        )
        rows = cursor.fetchall()
        conn.close()

        self.assertEqual(len(rows), 0)

    def test_escalated_status_set_on_overdue_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_photo_catalog.db")
            audit_path = os.path.join(tmp_dir, "assignment_audit.jsonl")

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    centroid BLOB,
                    n_confirmed INTEGER DEFAULT 0,
                    baseline_cohesion REAL DEFAULT 1.0,
                    current_cohesion REAL DEFAULT 1.0,
                    created_at TEXT,
                    last_updated TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE faces (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER,
                    encoding BLOB,
                    person_id INTEGER,
                    assignment_status TEXT,
                    review_candidate_ids TEXT,
                    review_candidate_scores TEXT,
                    review_margin REAL,
                    review_best_similarity REAL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE review_queue (
                    face_id INTEGER PRIMARY KEY,
                    encoding BLOB NOT NULL,
                    candidate_ids TEXT,
                    candidate_scores TEXT,
                    margin REAL,
                    best_similarity REAL,
                    enqueued_batch INTEGER DEFAULT 0,
                    batch_id TEXT,
                    created_at TEXT
                )
                """
            )

            emb_a = encode_embedding(np.array([1.0, 0.0], dtype=np.float32))
            emb_b = encode_embedding(np.array([0.0, 1.0], dtype=np.float32))

            # One unassigned orphan ensures main() reaches Phase 3.
            cursor.execute(
                "INSERT INTO faces (id, image_id, encoding, person_id, assignment_status) VALUES (?, ?, ?, ?, ?)",
                (1, 1, emb_a, -1, None),
            )
            # Two review-queue faces: one very old (should escalate), one recent (sets batch counter to 20).
            cursor.execute(
                "INSERT INTO faces (id, image_id, encoding, person_id, assignment_status) VALUES (?, ?, ?, ?, ?)",
                (2, 2, emb_b, None, "REVIEW"),
            )
            cursor.execute(
                "INSERT INTO faces (id, image_id, encoding, person_id, assignment_status) VALUES (?, ?, ?, ?, ?)",
                (3, 3, emb_b, None, "REVIEW"),
            )
            cursor.execute(
                "INSERT INTO review_queue (face_id, encoding, enqueued_batch, batch_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (2, emb_b, 0, "old", "2026-01-01T00:00:00"),
            )
            cursor.execute(
                "INSERT INTO review_queue (face_id, encoding, enqueued_batch, batch_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (3, emb_b, 19, "recent", "2026-01-01T00:00:00"),
            )
            conn.commit()
            conn.close()

            config = {
                "db_path": db_path,
                "assignment_audit_log_path": audit_path,
                "review_max_age_batches": 10,
            }
            module, _ = load_cluster_module_with_config("cluster_module_overdue", config)
            module.main()
            logger = module.logging.getLogger("face_assignment_audit")
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT assignment_status FROM faces WHERE id = 2")
            status = cursor.fetchone()[0]
            conn.close()

            self.assertEqual(status, "ESCALATED")

    def test_config_threshold_loading_nested_block(self):
        nested_cfg = {
            "assignment": {
                "high_threshold": 0.81,
                "low_threshold": 0.61,
                "margin_threshold": 0.13,
                "calibrated": False,
            }
        }
        module, output = load_cluster_module_with_config("cluster_module_nested_cfg", nested_cfg)
        assigner = module.ConfidenceGatedAssigner(
            {
                "high_threshold": module.HIGH_THRESHOLD,
                "low_threshold": module.LOW_THRESHOLD,
                "margin_threshold": module.MARGIN_THRESHOLD,
            }
        )

        self.assertAlmostEqual(assigner.high_thresh, 0.81)
        self.assertAlmostEqual(assigner.low_thresh, 0.61)
        self.assertAlmostEqual(assigner.margin_thresh, 0.13)
        self.assertIn("not calibrated", output)

    def test_config_threshold_loading_top_level_fallback(self):
        top_level_cfg = {
            "high_threshold": 0.79,
            "low_threshold": 0.58,
            "margin_threshold": 0.11,
        }
        module, _ = load_cluster_module_with_config("cluster_module_top_level_cfg", top_level_cfg)
        assigner = module.ConfidenceGatedAssigner(
            {
                "high_threshold": module.HIGH_THRESHOLD,
                "low_threshold": module.LOW_THRESHOLD,
                "margin_threshold": module.MARGIN_THRESHOLD,
            }
        )

        self.assertAlmostEqual(assigner.high_thresh, 0.79)
        self.assertAlmostEqual(assigner.low_thresh, 0.58)
        self.assertAlmostEqual(assigner.margin_thresh, 0.11)


if __name__ == "__main__":
    unittest.main()
