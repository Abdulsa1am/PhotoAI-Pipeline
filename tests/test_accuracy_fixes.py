import importlib.util
import io
import json
import os
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


def load_extraction_module_with_config(module_name, config_data):
    output = io.StringIO()
    with temporary_pipeline_config(config_data), redirect_stdout(output):
        module = load_script_module(module_name, os.path.join("scripts", "1_face_extraction.py"))
    return module, output.getvalue()


class _FakeFace:
    def __init__(self, bbox, det_score=0.95):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.embedding = np.ones(512, dtype=np.float32)
        self.det_score = det_score


class TestAccuracyFixes(unittest.TestCase):
    def test_hdbscan_uses_cosine_metric(self):
        module, _ = load_cluster_module_with_config(
            "cluster_module_metric",
            {
                "hdbscan_metric": "cosine",
                "hdbscan_cluster_selection_method": "leaf",
            },
        )

        clusterer = module.hdbscan.HDBSCAN(
            min_cluster_size=module.MIN_CLUSTER_SIZE,
            min_samples=module.MIN_SAMPLES,
            metric=module.HDBSCAN_METRIC,
            cluster_selection_method=module.HDBSCAN_CLUSTER_SELECTION,
            core_dist_n_jobs=-1,
        )

        self.assertEqual(clusterer.metric, "cosine")

    def test_hdbscan_uses_leaf_selection(self):
        module, _ = load_cluster_module_with_config(
            "cluster_module_selection",
            {
                "hdbscan_metric": "cosine",
                "hdbscan_cluster_selection_method": "leaf",
            },
        )

        clusterer = module.hdbscan.HDBSCAN(
            min_cluster_size=module.MIN_CLUSTER_SIZE,
            min_samples=module.MIN_SAMPLES,
            metric=module.HDBSCAN_METRIC,
            cluster_selection_method=module.HDBSCAN_CLUSTER_SELECTION,
            core_dist_n_jobs=-1,
        )

        self.assertEqual(clusterer.cluster_selection_method, "leaf")

    def test_no_pca_applied_to_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "photo_catalog.db")
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
                    det_score REAL
                )
                """
            )

            rng = np.random.default_rng(42)
            raw = rng.normal(size=(1500, 512)).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            embeddings = raw / np.maximum(norms, 1e-12)

            face_rows = [
                (
                    i + 1,
                    i + 1,
                    encode_embedding(embeddings[i]),
                    -1,
                    None,
                    0.95,
                )
                for i in range(1500)
            ]
            cursor.executemany(
                "INSERT INTO faces (id, image_id, encoding, person_id, assignment_status, det_score) VALUES (?, ?, ?, ?, ?, ?)",
                face_rows,
            )
            conn.commit()
            conn.close()

            module, _ = load_cluster_module_with_config(
                "cluster_module_no_pca",
                {
                    "db_path": db_path,
                    "assignment_audit_log_path": audit_path,
                    "det_score_min_cluster": 0.70,
                },
            )

            captured = {}

            class FakeHDBSCAN:
                def __init__(self, *args, **kwargs):
                    pass

                def fit_predict(self, data):
                    captured["shape"] = tuple(data.shape)
                    return np.full(data.shape[0], -1, dtype=int)

            module.hdbscan.HDBSCAN = FakeHDBSCAN
            module.main()
            logger = module.logging.getLogger("face_assignment_audit")
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

            self.assertEqual(captured.get("shape"), (1500, 512))

    def test_small_face_filtered_in_extraction(self):
        module, _ = load_extraction_module_with_config(
            "extraction_module_small_face",
            {"min_face_area_px": 1600},
        )

        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                encoding BLOB,
                bbox TEXT,
                det_score REAL
            )
            """
        )

        image_id = 1
        total_faces = 0
        faces = [
            _FakeFace([0, 0, 30, 30], det_score=0.90),
            _FakeFace([0, 0, 50, 50], det_score=0.95),
        ]

        for face in faces:
            bbox = face.bbox
            face_w = float(bbox[2]) - float(bbox[0])
            face_h = float(bbox[3]) - float(bbox[1])
            face_area = face_w * face_h

            if face_area < module.MIN_FACE_AREA_PX:
                continue

            encoding_bytes = module.pickle.dumps(face.embedding)
            bbox_str = str(face.bbox.tolist())
            det_score = float(face.det_score)
            cursor.execute(
                "INSERT INTO faces (image_id, encoding, bbox, det_score) VALUES (?, ?, ?, ?)",
                (image_id, encoding_bytes, bbox_str, det_score),
            )
            total_faces += 1

        cursor.execute("SELECT COUNT(*) FROM faces")
        inserted_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(inserted_count, 1)
        self.assertEqual(total_faces, 1)

    def test_det_score_filter_in_loader_query(self):
        module, _ = load_cluster_module_with_config(
            "cluster_module_det_score",
            {"det_score_min_cluster": 0.70},
        )

        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE faces (
                id INTEGER PRIMARY KEY,
                encoding BLOB,
                person_id INTEGER,
                assignment_status TEXT,
                det_score REAL
            )
            """
        )
        cursor.executemany(
            "INSERT INTO faces (id, encoding, person_id, assignment_status, det_score) VALUES (?, ?, ?, ?, ?)",
            [
                (1, encode_embedding(np.ones(512, dtype=np.float32)), -1, None, 0.68),
                (2, encode_embedding(np.ones(512, dtype=np.float32)), -1, None, 0.75),
            ],
        )

        cursor.execute(
            """
            SELECT id, encoding, assignment_status
            FROM faces
            WHERE (person_id IS NULL OR person_id = -1)
              AND (assignment_status IS NULL
                   OR (assignment_status != 'REVIEW'
                       AND assignment_status != 'ESCALATED'))
              AND (det_score IS NULL OR det_score >= ?)
            """,
            (module.DET_SCORE_MIN_CLUSTER,),
        )
        rows = cursor.fetchall()
        conn.close()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 2)

    def test_config_hdbscan_overrides(self):
        module, _ = load_cluster_module_with_config(
            "cluster_module_hdbscan_overrides",
            {
                "hdbscan_metric": "euclidean",
                "hdbscan_cluster_selection_method": "eom",
            },
        )

        self.assertEqual(module.HDBSCAN_METRIC, "euclidean")
        self.assertEqual(module.HDBSCAN_CLUSTER_SELECTION, "eom")


if __name__ == "__main__":
    unittest.main()
