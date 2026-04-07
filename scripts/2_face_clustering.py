"""
Script 2: Persistent Face Clustering (Confidence-Gated + HDBSCAN)

This script functions incrementally with a three-layer confidence gate:
1. Pulls new/unassigned faces from DB.
2. Assigns high-confidence matches to established people.
3. Defers ambiguous matches to a review queue.
4. Clusters only true new-identity candidates with HDBSCAN.
"""

import os
import sys
import sqlite3
import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import List, Optional
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from embedding_codec import encode_embedding, decode_embedding

try:
    import hdbscan
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("[ERROR] hdbscan/scikit-learn not installed")
    print("Install: pip install hdbscan scikit-learn")
    sys.exit(1)

# ============ CONFIGURATION ============
DB_PATH = r"D:\PhotoAI\photo_catalog.db"
MIN_CLUSTER_SIZE = 8
MIN_SAMPLES = 3
DET_SCORE_MIN_CLUSTER = 0.70
HDBSCAN_METRIC = "cosine"
HDBSCAN_CLUSTER_SELECTION = "leaf"
HIGH_THRESHOLD = 0.72
LOW_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.08
AUDIT_LOG_PATH = r"D:\PhotoAI\logs\assignment_audit.jsonl"
REVIEW_MAX_SIZE = 5000
REVIEW_MAX_AGE_BATCHES = 10

# ---- Load overrides from pipeline_config.json ----
_cfg_path = os.path.join(PROJECT_ROOT, "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    DB_PATH          = _cfg.get("db_path", DB_PATH)
    MIN_CLUSTER_SIZE = int(_cfg.get("min_cluster_size", MIN_CLUSTER_SIZE))
    MIN_SAMPLES      = int(_cfg.get("min_samples", MIN_SAMPLES))
    DET_SCORE_MIN_CLUSTER = float(
        _cfg.get("det_score_min_cluster", DET_SCORE_MIN_CLUSTER)
    )
    HDBSCAN_METRIC = _cfg.get("hdbscan_metric", "cosine")
    HDBSCAN_CLUSTER_SELECTION = _cfg.get(
        "hdbscan_cluster_selection_method", "leaf"
    )
    _assign_cfg = _cfg.get("assignment", {})
    HIGH_THRESHOLD   = float(_assign_cfg.get("high_threshold", _cfg.get("high_threshold", HIGH_THRESHOLD)))
    LOW_THRESHOLD    = float(_assign_cfg.get("low_threshold", _cfg.get("low_threshold", LOW_THRESHOLD)))
    MARGIN_THRESHOLD = float(_assign_cfg.get("margin_threshold", _cfg.get("margin_threshold", MARGIN_THRESHOLD)))

    # Read calibration flag and warn if not calibrated
    _calibrated = _assign_cfg.get("calibrated", _cfg.get("calibrated", False))
    if not _calibrated:
        print("[WARNING] Assignment thresholds are not calibrated on labeled data.")
        print("[WARNING] Running with default thresholds. False merge rate is unknown.")
        print("[WARNING] Run threshold calibration and set calibrated=true in config.")

    AUDIT_LOG_PATH   = _cfg.get("assignment_audit_log_path", AUDIT_LOG_PATH)
    REVIEW_MAX_SIZE  = int(_cfg.get("review_max_size", REVIEW_MAX_SIZE))
    REVIEW_MAX_AGE_BATCHES = int(_cfg.get("review_max_age_batches", REVIEW_MAX_AGE_BATCHES))
# =======================================


@dataclass
class AssignmentResult:
    status: str
    person_id: Optional[int]
    best_similarity: float
    margin: float
    candidate_ids: List[int]
    candidate_scores: List[float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    batch_id: str = ""


@dataclass
class IdentityRecord:
    person_id: int
    centroid: np.ndarray
    n_confirmed: int = 0
    baseline_cohesion: float = 1.0
    current_cohesion: float = 1.0
    created_at: str = ""
    last_updated: str = ""


class ConfidenceGatedAssigner:
    def __init__(self, config: dict):
        self.high_thresh = float(config.get("high_threshold", HIGH_THRESHOLD))
        self.low_thresh = float(config.get("low_threshold", LOW_THRESHOLD))
        self.margin_thresh = float(config.get("margin_threshold", MARGIN_THRESHOLD))

    def assign(self, embedding: np.ndarray, identities: List[IdentityRecord], batch_id: str = "") -> AssignmentResult:
        if not identities:
            return AssignmentResult(
                status="NEW_IDENTITY",
                person_id=None,
                best_similarity=0.0,
                margin=1.0,
                candidate_ids=[],
                candidate_scores=[],
                batch_id=batch_id,
            )

        centroids = np.stack([i.centroid for i in identities])
        similarities = cosine_similarity(embedding.reshape(1, -1), centroids).flatten()

        top2_indices = np.argsort(similarities)[-2:][::-1]
        best_idx = int(top2_indices[0])
        best_score = float(similarities[best_idx])
        second_score = float(similarities[top2_indices[1]]) if len(top2_indices) > 1 else 0.0
        margin = best_score - second_score

        candidate_ids = [identities[i].person_id for i in top2_indices]
        candidate_scores = [float(similarities[i]) for i in top2_indices]

        if best_score >= self.high_thresh and margin >= self.margin_thresh:
            return AssignmentResult(
                status="ASSIGNED",
                person_id=identities[best_idx].person_id,
                best_similarity=best_score,
                margin=margin,
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                batch_id=batch_id,
            )

        if best_score >= self.low_thresh:
            return AssignmentResult(
                status="REVIEW",
                person_id=None,
                best_similarity=best_score,
                margin=margin,
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                batch_id=batch_id,
            )

        return AssignmentResult(
            status="NEW_IDENTITY",
            person_id=None,
            best_similarity=best_score,
            margin=margin,
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            batch_id=batch_id,
        )


class CentroidManager:
    EMA_ALPHA_STABLE = 0.05
    EMA_ALPHA_YOUNG = 0.15

    def __init__(self, high_threshold: float, margin_threshold: float):
        self.high_threshold = high_threshold
        self.margin_threshold = margin_threshold

    def update(self, identity: IdentityRecord, new_embedding: np.ndarray, result: AssignmentResult) -> None:
        if result.status != "ASSIGNED":
            return
        if result.margin < self.margin_threshold or result.best_similarity < self.high_threshold:
            return

        alpha = self.EMA_ALPHA_YOUNG if identity.n_confirmed < 10 else self.EMA_ALPHA_STABLE
        updated = alpha * new_embedding + (1 - alpha) * identity.centroid
        updated_norm = np.linalg.norm(updated)
        if updated_norm > 0:
            updated = updated / updated_norm
        identity.centroid = updated
        identity.n_confirmed += 1
        identity.last_updated = datetime.utcnow().isoformat()

        new_cohesion = float(cosine_similarity(new_embedding.reshape(1, -1), identity.centroid.reshape(1, -1))[0][0])
        identity.current_cohesion = 0.1 * new_cohesion + 0.9 * identity.current_cohesion
        if identity.n_confirmed <= 1:
            identity.baseline_cohesion = identity.current_cohesion


class AssignmentAuditLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("face_assignment_audit")
        self._logger.handlers = []
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def log(self, face_id: int, result: AssignmentResult, identity: Optional[IdentityRecord], was_graduated: bool = False) -> None:
        record = {
            "face_id": str(face_id),
            "batch_id": result.batch_id,
            "timestamp": result.timestamp.isoformat(),
            "status": result.status,
            "person_id": result.person_id,
            "best_similarity": round(result.best_similarity, 4),
            "margin": round(result.margin, 4),
            "candidate_ids": result.candidate_ids,
            "candidate_scores": [round(s, 4) for s in result.candidate_scores],
            "centroid_age": identity.n_confirmed if identity else 0,
            "centroid_cohesion": round(identity.current_cohesion, 4) if identity else None,
            "was_graduated": was_graduated,
        }
        self._logger.info(json.dumps(record))


class ClusterHealthMonitor:
    DEGRADATION_ALERT_THRESHOLD = 0.20

    def scan(self, identities: List[IdentityRecord]) -> List[dict]:
        alerts = []
        for identity in identities:
            if identity.n_confirmed < 5:
                continue
            degradation = (identity.baseline_cohesion - identity.current_cohesion) / max(identity.baseline_cohesion, 1e-9)
            if degradation > self.DEGRADATION_ALERT_THRESHOLD:
                alerts.append({
                    "person_id": identity.person_id,
                    "baseline_cohesion": round(identity.baseline_cohesion, 4),
                    "current_cohesion": round(identity.current_cohesion, 4),
                    "degradation_pct": round(degradation * 100, 1),
                    "n_confirmed": identity.n_confirmed,
                })
        return alerts


def _build_hdbscan_clusterer() -> "hdbscan.HDBSCAN":
    kwargs = {
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "metric": HDBSCAN_METRIC,
        "cluster_selection_method": HDBSCAN_CLUSTER_SELECTION,
        "core_dist_n_jobs": -1,
    }

    # hdbscan's default BallTree path rejects metric='cosine'; generic works.
    if str(HDBSCAN_METRIC).strip().lower() == "cosine":
        kwargs["algorithm"] = "generic"

    return hdbscan.HDBSCAN(**kwargs)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(people)")
    people_cols = {row[1] for row in cursor.fetchall()}
    if "n_confirmed" not in people_cols:
        cursor.execute("ALTER TABLE people ADD COLUMN n_confirmed INTEGER DEFAULT 0")
    if "baseline_cohesion" not in people_cols:
        cursor.execute("ALTER TABLE people ADD COLUMN baseline_cohesion REAL DEFAULT 1.0")
    if "current_cohesion" not in people_cols:
        cursor.execute("ALTER TABLE people ADD COLUMN current_cohesion REAL DEFAULT 1.0")
    if "created_at" not in people_cols:
        cursor.execute("ALTER TABLE people ADD COLUMN created_at TEXT")
    if "last_updated" not in people_cols:
        cursor.execute("ALTER TABLE people ADD COLUMN last_updated TEXT")

    cursor.execute("PRAGMA table_info(faces)")
    face_cols = {row[1] for row in cursor.fetchall()}
    if "det_score" not in face_cols:
        cursor.execute("ALTER TABLE faces ADD COLUMN det_score REAL")
    if "person_id" not in face_cols:
        cursor.execute("ALTER TABLE faces ADD COLUMN person_id INTEGER")
    if "assignment_status" not in face_cols:
        cursor.execute("ALTER TABLE faces ADD COLUMN assignment_status TEXT")
    if "review_candidate_ids" not in face_cols:
        cursor.execute("ALTER TABLE faces ADD COLUMN review_candidate_ids TEXT")
    if "review_candidate_scores" not in face_cols:
        cursor.execute("ALTER TABLE faces ADD COLUMN review_candidate_scores TEXT")
    if "review_margin" not in face_cols:
        cursor.execute("ALTER TABLE faces ADD COLUMN review_margin REAL")
    if "review_best_similarity" not in face_cols:
        cursor.execute("ALTER TABLE faces ADD COLUMN review_best_similarity REAL")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS review_queue (
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
    conn.commit()


def _load_identities(cursor: sqlite3.Cursor) -> List[IdentityRecord]:
    cursor.execute(
        """
        SELECT id, centroid, COALESCE(n_confirmed, 0), COALESCE(baseline_cohesion, 1.0),
               COALESCE(current_cohesion, 1.0), COALESCE(created_at, ''), COALESCE(last_updated, '')
        FROM people
        WHERE centroid IS NOT NULL
        """
    )
    rows = cursor.fetchall()
    identities = []
    for row in rows:
        centroid, upgraded_blob, was_upgraded = decode_embedding(row[1], return_upgraded_blob=True)
        if was_upgraded:
            cursor.execute("UPDATE people SET centroid = ? WHERE id = ?", (upgraded_blob, row[0]))
        identities.append(
            IdentityRecord(
                person_id=row[0],
                centroid=normalize(np.array([centroid]), norm='l2')[0],
                n_confirmed=int(row[2]),
                baseline_cohesion=float(row[3]),
                current_cohesion=float(row[4]),
                created_at=row[5],
                last_updated=row[6],
            )
        )
    return identities


def _persist_identity(cursor: sqlite3.Cursor, identity: IdentityRecord) -> None:
    cursor.execute(
        """
        UPDATE people
        SET centroid = ?, n_confirmed = ?, baseline_cohesion = ?, current_cohesion = ?, last_updated = ?
        WHERE id = ?
        """,
        (
            encode_embedding(identity.centroid),
            identity.n_confirmed,
            identity.baseline_cohesion,
            identity.current_cohesion,
            identity.last_updated,
            identity.person_id,
        ),
    )


def _upsert_review(cursor: sqlite3.Cursor, face_id: int, encoding: np.ndarray, result: AssignmentResult, batch_counter: int) -> None:
    cursor.execute(
        """
        INSERT INTO review_queue (face_id, encoding, candidate_ids, candidate_scores, margin, best_similarity, enqueued_batch, batch_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(face_id) DO UPDATE SET
            encoding = excluded.encoding,
            candidate_ids = excluded.candidate_ids,
            candidate_scores = excluded.candidate_scores,
            margin = excluded.margin,
            best_similarity = excluded.best_similarity,
            batch_id = excluded.batch_id
        """,
        (
            face_id,
            encode_embedding(encoding),
            json.dumps(result.candidate_ids),
            json.dumps(result.candidate_scores),
            result.margin,
            result.best_similarity,
            batch_counter,
            result.batch_id,
            datetime.utcnow().isoformat(),
        ),
    )
def main():
    print("=" * 60)
    print("  Script 2: Incremental Face Clustering")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print("  Database not found. Run Script 1 first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    _ensure_schema(conn)

    batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cursor.execute("SELECT COALESCE(MAX(enqueued_batch), 0) FROM review_queue")
    current_batch_counter = int(cursor.fetchone()[0]) + 1

    assigner = ConfidenceGatedAssigner(
        {
            "high_threshold": HIGH_THRESHOLD,
            "low_threshold": LOW_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
        }
    )
    centroid_manager = CentroidManager(HIGH_THRESHOLD, MARGIN_THRESHOLD)
    audit_logger = AssignmentAuditLogger(AUDIT_LOG_PATH)

    identities = _load_identities(cursor)

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
                (DET_SCORE_MIN_CLUSTER,),
    )
    unassigned_rows = cursor.fetchall()

    escalated_in_batch = sum(1 for _, _, status in unassigned_rows if status == 'ESCALATED')
    assert escalated_in_batch == 0, "ESCALATED faces leaked into unassigned loader batch"
    print("  Safety check: 0 ESCALATED faces in unassigned batch.")

    if not unassigned_rows:
        print("  ✓ No new faces to cluster.")
        conn.close()
        return

    unassigned_ids = [r[0] for r in unassigned_rows]
    upgraded_unassigned = []
    unassigned_encodings = []
    for r in unassigned_rows:
        emb, upgraded_blob, was_upgraded = decode_embedding(r[1], return_upgraded_blob=True)
        unassigned_encodings.append(emb)
        if was_upgraded:
            upgraded_unassigned.append((upgraded_blob, r[0]))
    if upgraded_unassigned:
        cursor.executemany("UPDATE faces SET encoding = ? WHERE id = ?", upgraded_unassigned)
    unassigned_encodings = np.array(unassigned_encodings)
    unassigned_norm = normalize(unassigned_encodings, norm='l2')

    print(f"  Loaded {len(unassigned_rows)} unassigned/orphan faces.")

    new_identity_idx = []
    assigned_count = 0
    review_count = 0

    print("  Phase 1: Confidence-gated assignment to existing identities...")
    id_map = {identity.person_id: identity for identity in identities}

    for i, embedding in enumerate(unassigned_norm):
        face_id = unassigned_ids[i]
        result = assigner.assign(embedding, identities, batch_id=batch_id)

        if result.status == "ASSIGNED" and result.person_id in id_map:
            identity = id_map[result.person_id]
            centroid_manager.update(identity, embedding, result)
            _persist_identity(cursor, identity)
            cursor.execute(
                """
                UPDATE faces
                SET person_id = ?, assignment_status = 'ASSIGNED',
                    review_candidate_ids = NULL, review_candidate_scores = NULL,
                    review_margin = NULL, review_best_similarity = NULL
                WHERE id = ?
                """,
                (result.person_id, face_id),
            )
            cursor.execute("DELETE FROM review_queue WHERE face_id = ?", (face_id,))
            audit_logger.log(face_id, result, identity)
            assigned_count += 1
        elif result.status == "REVIEW":
            cursor.execute(
                """
                UPDATE faces
                SET person_id = NULL, assignment_status = 'REVIEW',
                    review_candidate_ids = ?, review_candidate_scores = ?,
                    review_margin = ?, review_best_similarity = ?
                WHERE id = ?
                """,
                (
                    json.dumps(result.candidate_ids),
                    json.dumps(result.candidate_scores),
                    result.margin,
                    result.best_similarity,
                    face_id,
                ),
            )
            _upsert_review(cursor, face_id, embedding, result, current_batch_counter)
            audit_logger.log(face_id, result, None)
            review_count += 1
        else:
            new_identity_idx.append(i)

    conn.commit()
    print(f"    -> Assigned: {assigned_count}")
    print(f"    -> Deferred to review: {review_count}")

    # ---- Phase 2: Differential HDBSCAN ----
    if new_identity_idx:
        print(f"  Phase 2: Clustering {len(new_identity_idx)} new-identity candidates with HDBSCAN...")
        
        # HDBSCAN needs at least a handful of samples to avoid blowing up geometrically
        if len(new_identity_idx) < MIN_CLUSTER_SIZE:
            print(f"    -> Not enough unfamiliar faces to form a new valid cluster. Marked as Unknown.")
            for i in new_identity_idx:
                cursor.execute(
                    "UPDATE faces SET person_id = -1, assignment_status = 'NEW_IDENTITY' WHERE id = ?",
                    (unassigned_ids[i],),
                )
            conn.commit()
            new_identity_idx = []

        if new_identity_idx:
            leftover_norm = unassigned_norm[new_identity_idx]

            cluster_data = leftover_norm  # always use full 512D
            
            start = time.time()
            clusterer = _build_hdbscan_clusterer()
            cluster_input = cluster_data
            if (
                str(getattr(clusterer, "metric", "")).strip().lower() == "cosine"
                and str(getattr(clusterer, "algorithm", "")).strip().lower() == "generic"
            ):
                # hdbscan generic cosine path expects float64 in mst_linkage_core.
                cluster_input = np.asarray(cluster_data, dtype=np.float64)

            labels = clusterer.fit_predict(cluster_input)
        
            unique_labels = [l for l in set(labels) if l != -1]
        
            new_people_count = 0
            for label in unique_labels:
                mask = labels == label
                centroid = leftover_norm[mask].mean(axis=0)
                centroid /= np.linalg.norm(centroid)

                now_iso = datetime.utcnow().isoformat()
                c_blob = encode_embedding(centroid)
                cluster_size = int(np.sum(mask))
                cursor.execute(
                    """
                    INSERT INTO people (centroid, n_confirmed, baseline_cohesion, current_cohesion, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (c_blob, cluster_size, 1.0, 1.0, now_iso, now_iso),
                )
                new_p_id = cursor.lastrowid
                identities.append(
                    IdentityRecord(
                        person_id=new_p_id,
                        centroid=centroid,
                        n_confirmed=cluster_size,
                        baseline_cohesion=1.0,
                        current_cohesion=1.0,
                        created_at=now_iso,
                        last_updated=now_iso,
                    )
                )
                id_map[new_p_id] = identities[-1]
                new_people_count += 1

                face_idxs = np.where(labels == label)[0]
                for f_idx in face_idxs:
                    actual_id = unassigned_ids[new_identity_idx[f_idx]]
                    cursor.execute(
                        "UPDATE faces SET person_id = ?, assignment_status = 'ASSIGNED' WHERE id = ?",
                        (new_p_id, actual_id),
                    )
                
            noise_idx = np.where(labels == -1)[0]
            for n_idx in noise_idx:
                actual_id = unassigned_ids[new_identity_idx[n_idx]]
                cursor.execute(
                    "UPDATE faces SET person_id = -1, assignment_status = 'NEW_IDENTITY' WHERE id = ?",
                    (actual_id,),
                )

            conn.commit()

            elapsed = time.time() - start
            print(f"    -> Discovered {new_people_count} new unique people in {elapsed:.1f}s.")
            print(f"    -> {len(noise_idx)} faces marked as Unknown (Orphans).")

    print("  Phase 3: Re-evaluating deferred review queue...")
    cursor.execute(
        "SELECT face_id, encoding, enqueued_batch FROM review_queue ORDER BY enqueued_batch ASC LIMIT ?",
        (REVIEW_MAX_SIZE,),
    )
    queued_rows = cursor.fetchall()

    graduated_count = 0
    escalated_count = 0
    still_pending = deque()

    for face_id, enc_blob, enqueued_batch in queued_rows:
        age = current_batch_counter - int(enqueued_batch)
        embedding, upgraded_blob, was_upgraded = decode_embedding(enc_blob, return_upgraded_blob=True)
        if was_upgraded:
            cursor.execute("UPDATE review_queue SET encoding = ? WHERE face_id = ?", (upgraded_blob, face_id))
        embedding = normalize(np.array([embedding]), norm='l2')[0]

        if age > REVIEW_MAX_AGE_BATCHES:
            cursor.execute(
                "UPDATE faces SET person_id = -1, assignment_status = 'ESCALATED' WHERE id = ?",
                (face_id,),
            )
            cursor.execute("DELETE FROM review_queue WHERE face_id = ?", (face_id,))
            escalated_count += 1
            continue

        result = assigner.assign(embedding, identities, batch_id=batch_id)
        if result.status == "ASSIGNED" and result.person_id in id_map:
            identity = id_map[result.person_id]
            centroid_manager.update(identity, embedding, result)
            _persist_identity(cursor, identity)
            cursor.execute(
                """
                UPDATE faces
                SET person_id = ?, assignment_status = 'ASSIGNED',
                    review_candidate_ids = NULL, review_candidate_scores = NULL,
                    review_margin = NULL, review_best_similarity = NULL
                WHERE id = ?
                """,
                (result.person_id, face_id),
            )
            cursor.execute("DELETE FROM review_queue WHERE face_id = ?", (face_id,))
            audit_logger.log(face_id, result, identity, was_graduated=True)
            graduated_count += 1
        else:
            still_pending.append(face_id)

    conn.commit()
    print(f"    -> Graduated from review: {graduated_count}")
    print(f"    -> Escalated after max age: {escalated_count}")
    print(f"    -> Still pending: {len(still_pending)}")

    alerts = ClusterHealthMonitor().scan(identities)
    if alerts:
        print(f"  [ALERT] {len(alerts)} identity clusters show cohesion degradation > 20%")
    else:
        print("  Cluster health check: no degradation alerts")

    conn.close()
    print("\n  Cluster calculation complete!")

if __name__ == "__main__":
    main()