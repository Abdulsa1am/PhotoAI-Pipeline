# PhotoAI Pipeline — Incremental Face Assignment: Improvement Plan

> **Agent:** Senior ML Engineer & Python Refactoring Assistant
> **Domain:** Computer Vision · Incremental Clustering · Identity Management
> **Target System:** PhotoAI Face Assignment Pipeline
> **Status:** Active Redesign Plan · April 2026

---

## Mission Brief

The current pipeline assigns new faces to existing identities using a single similarity threshold against stored centroids. This strategy is greedy, irreversible, and propagates assignment errors into archive folder structures where they compound across batches. This document is the complete improvement plan: problem model, architectural redesign, implementation specs, and validation protocol.

---

## Phase 0 — Threat Model (What Exactly Is Breaking)

Before any code is written, the agent must understand the failure topology.

**Root cause chain:**

```
Greedy nearest-centroid assignment
        ↓
Mislabeled face assigned to wrong identity (false merge)
        ↓
Centroid updated with corrupted embedding
        ↓
Future faces compared against contaminated centroid
        ↓
More false merges (drift accelerates with batch size)
        ↓
Archive folder written with wrong person_id
        ↓
Irreversible structural error in output
```

**Three compounding failure modes:**

| Failure Mode | Mechanism | Observable Symptom |
|---|---|---|
| **Centroid contamination** | Mislabeled faces update the centroid mean | Identity clusters slowly migrate away from true face distribution |
| **Symmetry blindness** | No margin check between top-1 and top-2 candidates | Ambiguous faces (near two identities equally) get forced into one |
| **No corrective loop** | No audit trail, no re-evaluation path | Errors are invisible until archive folder inspection |

**Key insight:** A face with `best_score = 0.71` and `margin = 0.02` is *far more dangerous* than one with `best_score = 0.62` and `margin = 0.30`. The current system cannot distinguish them.

---

## Phase 1 — Architecture Redesign

### Decision: Replace Single-Threshold Assignment with a Three-Layer Confidence Gate

```
                    ┌─────────────────────────────────┐
                    │         INCOMING FACE            │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   Compute cosine similarity to  │
                    │   ALL existing centroids        │
                    │   → get top-2 scores + margin   │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  score >= HIGH_THRESH            │
                    │  AND margin >= MARGIN_THRESH?   │
                    └──────┬────────────────┬─────────┘
                          YES               NO
                           │                │
                           ▼                ▼
                    ┌──────────┐   ┌────────────────┐
                    │ ASSIGNED │   │score >= LOW_TH?│
                    │ → Archive│   └───┬────────┬───┘
                    │ → EMA    │      YES       NO
                    │   update │       │         │
                    └──────────┘       ▼         ▼
                                 ┌──────────┐ ┌─────────────┐
                                 │  REVIEW  │ │NEW_IDENTITY │
                                 │ → Staging│ │ → New       │
                                 │   queue  │ │   centroid  │
                                 └──────────┘ └─────────────┘
```

### New Data Structures

```python
from dataclasses import dataclass, field
from typing import Literal, List, Optional
from datetime import datetime
import numpy as np

@dataclass
class AssignmentResult:
    status: Literal["ASSIGNED", "REVIEW", "NEW_IDENTITY"]
    person_id: Optional[int]
    best_similarity: float
    margin: float                    # best_score - second_best_score
    candidate_ids: List[int]         # top-2 candidates
    candidate_scores: List[float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    batch_id: str = ""

@dataclass
class IdentityRecord:
    person_id: int
    centroid: np.ndarray
    n_confirmed: int = 0             # Only high-confidence assignments
    baseline_cohesion: float = 1.0
    current_cohesion: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AssignmentAuditRecord:
    face_id: str
    batch_id: str
    timestamp: datetime
    status: Literal["ASSIGNED", "REVIEW", "NEW_IDENTITY"]
    best_similarity: float
    margin: float
    person_id: Optional[int]
    candidate_ids: List[int]
    centroid_age: int                # n_confirmed at time of decision
    was_graduated: bool = False      # True if promoted from REVIEW later
    graduation_batch: Optional[str] = None
```

---

## Phase 2 — Core Implementation

### 2.1 — The Assignment Engine

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ConfidenceGatedAssigner:
    """
    Three-layer assignment gate.
    Replaces: single threshold nearest-centroid assignment.
    """

    def __init__(self, config: dict):
        self.HIGH_THRESH   = config.get("high_threshold", 0.72)
        self.LOW_THRESH    = config.get("low_threshold", 0.55)
        self.MARGIN_THRESH = config.get("margin_threshold", 0.08)

    def assign(
        self,
        embedding: np.ndarray,
        identities: list[IdentityRecord],
        batch_id: str = "",
    ) -> AssignmentResult:

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
        similarities = cosine_similarity(
            embedding.reshape(1, -1), centroids
        ).flatten()

        top2_indices = np.argsort(similarities)[-2:][::-1]
        best_score  = similarities[top2_indices[0]]
        second_score = similarities[top2_indices[1]] if len(top2_indices) > 1 else 0.0
        margin = float(best_score - second_score)

        # Layer 1: High-confidence, unambiguous
        if best_score >= self.HIGH_THRESH and margin >= self.MARGIN_THRESH:
            return AssignmentResult(
                status="ASSIGNED",
                person_id=identities[top2_indices[0]].person_id,
                best_similarity=float(best_score),
                margin=margin,
                candidate_ids=[identities[i].person_id for i in top2_indices],
                candidate_scores=[float(similarities[i]) for i in top2_indices],
                batch_id=batch_id,
            )

        # Layer 2: Mid-confidence or ambiguous — defer
        if best_score >= self.LOW_THRESH:
            return AssignmentResult(
                status="REVIEW",
                person_id=None,
                best_similarity=float(best_score),
                margin=margin,
                candidate_ids=[identities[i].person_id for i in top2_indices],
                candidate_scores=[float(similarities[i]) for i in top2_indices],
                batch_id=batch_id,
            )

        # Layer 3: No viable match — new identity
        return AssignmentResult(
            status="NEW_IDENTITY",
            person_id=None,
            best_similarity=float(best_score),
            margin=margin,
            candidate_ids=[identities[i].person_id for i in top2_indices],
            candidate_scores=[float(similarities[i]) for i in top2_indices],
            batch_id=batch_id,
        )
```

### 2.2 — Protected Centroid Update (EMA)

```python
class CentroidManager:
    """
    Manages centroid updates with EMA.
    CRITICAL: Only high-confidence, unambiguous faces update the centroid.
    """

    # Use a higher alpha for young identities (few confirmed faces)
    EMA_ALPHA_STABLE = 0.05   # for n_confirmed >= 10
    EMA_ALPHA_YOUNG  = 0.15   # for n_confirmed < 10

    def update(
        self,
        identity: IdentityRecord,
        new_embedding: np.ndarray,
        result: AssignmentResult,
    ) -> None:
        # Guard: never update on REVIEW or NEW_IDENTITY results
        if result.status != "ASSIGNED":
            return
        # Guard: only update if margin and score are genuinely strong
        if result.margin < 0.08 or result.best_similarity < 0.72:
            return

        alpha = (
            self.EMA_ALPHA_YOUNG
            if identity.n_confirmed < 10
            else self.EMA_ALPHA_STABLE
        )

        identity.centroid = (
            alpha * new_embedding + (1 - alpha) * identity.centroid
        )
        identity.n_confirmed += 1
        identity.last_updated = datetime.utcnow()

        # Update cohesion score (cosine sim of new embedding to updated centroid)
        new_cohesion = float(
            cosine_similarity(
                new_embedding.reshape(1, -1),
                identity.centroid.reshape(1, -1)
            )
        )
        # EMA of cohesion score itself
        identity.current_cohesion = (
            0.1 * new_cohesion + 0.9 * identity.current_cohesion
        )
```

### 2.3 — Review Queue with Deferred Graduation

```python
from collections import deque

class ReviewQueue:
    """
    Holds ambiguous faces for re-evaluation once centroids stabilize.
    Prevents the queue from growing unbounded with a max-size cap.
    """

    MAX_SIZE = 5000
    MAX_AGE_BATCHES = 10  # Discard if unresolved after N batches

    def __init__(self):
        self.pending: deque = deque(maxlen=self.MAX_SIZE)
        self.current_batch_counter = 0

    def enqueue(self, face_id: str, embedding: np.ndarray, result: AssignmentResult):
        self.pending.append({
            "face_id": face_id,
            "embedding": embedding,
            "original_result": result,
            "enqueued_batch": self.current_batch_counter,
        })

    def process(
        self,
        assigner: ConfidenceGatedAssigner,
        identities: list[IdentityRecord],
        batch_id: str,
    ) -> list[dict]:
        """
        Re-evaluate all pending faces against updated centroids.
        Returns a list of newly resolved assignments for audit logging.
        """
        self.current_batch_counter += 1
        resolved = []
        still_pending = deque()

        for item in self.pending:
            age = self.current_batch_counter - item["enqueued_batch"]
            if age > self.MAX_AGE_BATCHES:
                # Escalate to manual review or assign as new identity
                resolved.append({**item, "final_status": "ESCALATED"})
                continue

            new_result = assigner.assign(item["embedding"], identities, batch_id)

            if new_result.status == "ASSIGNED":
                resolved.append({
                    **item,
                    "final_result": new_result,
                    "final_status": "GRADUATED",
                })
            else:
                still_pending.append(item)

        self.pending = still_pending
        return resolved
```

---

## Phase 3 — Observability & Audit System

### 3.1 — Structured Audit Logger

```python
import json
import logging
from pathlib import Path

class AssignmentAuditLogger:
    """
    Writes structured JSONL audit records for every assignment decision.
    Enables post-processing analysis, model improvement, and drift detection.
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("face_assignment_audit")
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def log(
        self,
        face_id: str,
        result: AssignmentResult,
        identity: Optional[IdentityRecord],
    ) -> None:
        record = {
            "face_id": face_id,
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
        }
        self._logger.info(json.dumps(record))
```

### 3.2 — Cluster Health Monitor

```python
class ClusterHealthMonitor:
    """
    Tracks per-identity cohesion degradation.
    Flags identities where drift is statistically significant.
    """

    DEGRADATION_ALERT_THRESHOLD = 0.20  # 20% cohesion drop from baseline

    def scan(self, identities: list[IdentityRecord]) -> list[dict]:
        alerts = []
        for identity in identities:
            if identity.n_confirmed < 5:
                continue  # Too few samples to make a meaningful assessment

            degradation = (
                (identity.baseline_cohesion - identity.current_cohesion)
                / max(identity.baseline_cohesion, 1e-9)
            )

            if degradation > self.DEGRADATION_ALERT_THRESHOLD:
                alerts.append({
                    "person_id": identity.person_id,
                    "baseline_cohesion": round(identity.baseline_cohesion, 4),
                    "current_cohesion": round(identity.current_cohesion, 4),
                    "degradation_pct": round(degradation * 100, 1),
                    "n_confirmed": identity.n_confirmed,
                    "recommended_action": "RECHECK_RECENT_ASSIGNMENTS",
                })

        return alerts
```

---

## Phase 4 — Archive Strategy (Backward Compatibility)

**Guiding principle:** Existing person_id folders are immutable. The new pipeline only *adds* a staging layer for unresolved faces.

```
archive/
├── person_0001/           ← existing, untouched
├── person_0002/           ← existing, untouched
├── ...
└── _pending/              ← NEW: staging for REVIEW status faces
    ├── face_abc123.jpg
    ├── face_def456.jpg
    └── ...

logs/
└── assignment_audit.jsonl ← NEW: structured audit trail
```

**Rules:**

- `ASSIGNED` → written directly to `archive/person_XXXX/` (same as before)
- `REVIEW` → written to `archive/_pending/` with metadata sidecar JSON
- `NEW_IDENTITY` → new `archive/person_XXXX/` folder created
- Graduated faces from review queue → moved from `_pending/` to `archive/person_XXXX/`
- No existing folder is renamed, merged, or deleted without explicit human confirmation

---

## Phase 5 — Threshold Calibration Protocol

Thresholds are not magic constants — they must be calibrated on representative labeled data.

```python
def calibrate_thresholds(
    labeled_embeddings: list[tuple[np.ndarray, int]],   # (embedding, true_person_id)
    high_range=(0.65, 0.85),
    low_range=(0.45, 0.65),
    margin_range=(0.04, 0.16),
    steps=10,
) -> dict:
    """
    Sweep threshold space and find the Pareto-optimal point:
    minimum false merge rate given a maximum acceptable review queue fraction.
    """
    best_config = None
    best_false_merge_rate = 1.0

    for high in np.linspace(*high_range, steps):
        for low in np.linspace(*low_range, steps):
            for margin in np.linspace(*margin_range, steps):
                config = {
                    "high_threshold": high,
                    "low_threshold": low,
                    "margin_threshold": margin,
                }
                assigner = ConfidenceGatedAssigner(config)
                false_merges, review_fraction = evaluate(assigner, labeled_embeddings)

                if review_fraction <= 0.25 and false_merges < best_false_merge_rate:
                    best_false_merge_rate = false_merges
                    best_config = config

    return best_config
```

**Target operating point:** Review queue fraction ≤ 25% of batch, false merge rate ≤ 2%.

---

## Phase 6 — Testing & Validation Strategy

### Unit Tests

| Test | What to verify |
|---|---|
| `test_high_confidence_assign` | Score ≥ HIGH_THRESH and margin ≥ MARGIN_THRESH → status = ASSIGNED |
| `test_ambiguous_goes_to_review` | High score but small margin → status = REVIEW |
| `test_low_score_new_identity` | Score < LOW_THRESH → status = NEW_IDENTITY |
| `test_centroid_not_updated_on_review` | REVIEW result must not update any centroid |
| `test_ema_centroid_stability` | Inject 10% mislabeled faces; centroid shift < epsilon |
| `test_review_graduation` | After centroid stabilizes, REVIEW face resolves to ASSIGNED |

### Integration Test — Replay Test

```
1. Take a known-good labeled batch (ground truth person_ids available)
2. Run through OLD pipeline → record false merge rate, drift scores
3. Run through NEW pipeline with calibrated thresholds → record same metrics
4. Compare: false merge rate, review queue size, archive structure correctness
5. Accept if: false merges reduced ≥ 50%, review queue graduates ≥ 70% within 3 batches
```

### Monitoring in Production

Track these metrics per batch in a dashboard:

- `review_queue_size` — should not grow monotonically (signals thresholds too strict)
- `graduation_rate` — should be ≥ 70% per batch window
- `flagged_identities` — from ClusterHealthMonitor; spikes indicate a noisy batch
- `avg_margin` per batch — sudden drops signal distributional shift in inputs
- `centroid_age_p50` — median confirmed count per identity; low values indicate underspecified identities

---

## Phase 7 — Trade-off Summary

| Decision | Benefit | Trade-off | Mitigation |
|---|---|---|---|
| Margin-based gating | Eliminates most false merges | ~15–25% more faces enter review queue initially | Threshold calibration on representative data |
| EMA centroid updates | Prevents centroid contamination | Slower adaptation to legitimate appearance change (aging, hairstyle) | Use higher alpha (0.15) for young identities (`n_confirmed < 10`) |
| Deferred review queue | Self-healing, no full reprocessing needed | Increases peak memory; queue needs max-size cap | `deque(maxlen=5000)` cap + age-based escalation |
| Cluster health monitoring | Early drift detection per identity | Adds O(k) overhead per batch | Only run monitor every N batches in production |
| Conservative alpha = 0.05 | Stable, trustworthy centroids | New identities build slowly | Dynamic alpha by `n_confirmed` |
| `_pending/` staging folder | Zero data migration required | Graduates must be moved; requires cleanup job | Nightly graduation job in pipeline scheduler |

---

## Recommended Rollout Sequence

```
Week 1:  Calibrate thresholds on labeled holdout set
         → Produce calibration report with operating point selection

Week 2:  Deploy AssignmentAuditLogger + ClusterHealthMonitor (read-only)
         → No logic change yet; establish baseline metrics from logs

Week 3:  Replace assignment logic with ConfidenceGatedAssigner
         → Shadow mode: run old + new in parallel; compare outputs

Week 4:  Enable protected CentroidManager (EMA updates)
         → Monitor centroid drift metrics; compare with old baseline

Week 5:  Enable ReviewQueue + _pending/ staging
         → Monitor graduation rate; tune MAX_AGE_BATCHES

Week 6+: Full production deployment
         → Ongoing: ClusterHealthMonitor alerts + threshold recalibration quarterly
```

---

*Document generated by AI Engineering Agent — PhotoAI Pipeline Refactor Initiative, April 2026.*
