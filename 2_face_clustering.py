"""
Script 2: Persistent Face Clustering (HDBSCAN + KNN)

This script now functions incrementally:
1. Pulls new faces from DB.
2. Checks them against established `people` centroids (KNN).
3. If they don't match, runs HDBSCAN to discover *new* people.
"""

import os
import sys
import sqlite3
import pickle
import json
import time
import numpy as np

try:
    import hdbscan
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist
except ImportError:
    print("[ERROR] hdbscan/scikit-learn/scipy not installed")
    print("Install: pip install hdbscan scikit-learn scipy")
    sys.exit(1)

# ============ CONFIGURATION ============
DB_PATH = r"D:\PhotoAI\photo_catalog.db"
MIN_CLUSTER_SIZE = 4
MIN_SAMPLES = 2
MERGE_THRESHOLD = 0.40

# ---- Load overrides from pipeline_config.json ----
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    DB_PATH          = _cfg.get("db_path", DB_PATH)
    MIN_CLUSTER_SIZE = int(_cfg.get("min_cluster_size", MIN_CLUSTER_SIZE))
    MERGE_THRESHOLD  = float(_cfg.get("merge_threshold", MERGE_THRESHOLD))
# =======================================

def main():
    print("=" * 60)
    print("  Script 2: Incremental Face Clustering")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print("  Database not found. Run Script 1 first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---- 1. Load established people ----
    cursor.execute("SELECT id, centroid FROM people")
    people_rows = cursor.fetchall()
    
    people_ids = []
    people_centroids = []
    
    for row in people_rows:
        if row[1] is not None:
            people_ids.append(row[0])
            people_centroids.append(pickle.loads(row[1]))
            
    if people_centroids:
        people_centroids_norm = normalize(np.array(people_centroids), norm='l2')
    else:
        people_centroids_norm = np.empty((0, 512))

    # ---- 2. Fetch unassigned faces ----
    # person_id can be NULL (fresh face) or we can optionally wipe -1s to re-test them.
    # Wiping -1 allows orphans to be matched if a new person was introduced.
    cursor.execute('''
        SELECT id, encoding FROM faces 
        WHERE person_id IS NULL OR person_id = -1
    ''')
    unassigned_rows = cursor.fetchall()

    if not unassigned_rows:
        print("  ✓ No new faces to cluster.")
        conn.close()
        return

    unassigned_ids = [r[0] for r in unassigned_rows]
    unassigned_encodings = np.array([pickle.loads(r[1]) for r in unassigned_rows])
    unassigned_norm = normalize(unassigned_encodings, norm='l2')

    print(f"  Loaded {len(unassigned_rows)} unassigned/orphan faces.")

    leftover_idx = []
    assigned_count = 0

    # ---- Phase 1: Fast Alignment (KNN) ----
    if len(people_centroids_norm) > 0:
        print("  Phase 1: Assigning to existing people via Cosine Similarity...")
        
        # Calculate distances from every new face to every known centroid
        distances = cdist(unassigned_norm, people_centroids_norm, metric='cosine')
        
        # Find the single closest centroid for each face
        min_dist_idx = np.argmin(distances, axis=1)
        min_dists = distances[np.arange(len(distances)), min_dist_idx]
        
        for i in range(len(unassigned_norm)):
            if min_dists[i] <= MERGE_THRESHOLD:
                # Assign explicitly to established row
                p_id = people_ids[min_dist_idx[i]]
                cursor.execute("UPDATE faces SET person_id = ? WHERE id = ?", (p_id, unassigned_ids[i]))
                assigned_count += 1
            else:
                leftover_idx.append(i)
                
        conn.commit()
    else:
        leftover_idx = list(range(len(unassigned_norm)))

    if assigned_count > 0:
        print(f"    -> {assigned_count} faces instantly matched established profiles!")

    # ---- Phase 2: Differential HDBSCAN ----
    if leftover_idx:
        print(f"  Phase 2: Clustering {len(leftover_idx)} unfamiliar faces with HDBSCAN...")
        
        # HDBSCAN needs at least a handful of samples to avoid blowing up geometrically
        if len(leftover_idx) < MIN_CLUSTER_SIZE:
            print(f"    -> Not enough unfamiliar faces to form a new valid cluster. Marked as Unknown.")
            for i in leftover_idx:
                cursor.execute("UPDATE faces SET person_id = -1 WHERE id = ?", (unassigned_ids[i],))
            conn.commit()
            conn.close()
            return
            
        leftover_norm = unassigned_norm[leftover_idx]

        # Dimensionality Reduction
        if len(leftover_norm) > 1000:
            print("    -> Reducing dimensionality (PCA 512D -> 96D)...")
            pca = PCA(n_components=min(96, len(leftover_norm)))
            cluster_data = pca.fit_transform(leftover_norm)
        else:
            cluster_data = leftover_norm
            
        start = time.time()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method='eom',
            core_dist_n_jobs=-1
        )
        labels = clusterer.fit_predict(cluster_data)
        
        unique_labels = [l for l in set(labels) if l != -1]
        
        new_people_count = 0
        for label in unique_labels:
            mask = labels == label
            centroid = leftover_norm[mask].mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            
            # Create a brand new established person in the DB
            c_blob = pickle.dumps(centroid)
            cursor.execute("INSERT INTO people (centroid) VALUES (?)", (c_blob,))
            new_p_id = cursor.lastrowid
            new_people_count += 1
            
            # Map faces
            face_idxs = np.where(labels == label)[0]
            for f_idx in face_idxs:
                actual_id = unassigned_ids[leftover_idx[f_idx]]
                cursor.execute("UPDATE faces SET person_id = ? WHERE id = ?", (new_p_id, actual_id))
                
        # Orphans (Noise)
        noise_idx = np.where(labels == -1)[0]
        for n_idx in noise_idx:
            actual_id = unassigned_ids[leftover_idx[n_idx]]
            cursor.execute("UPDATE faces SET person_id = -1 WHERE id = ?", (actual_id,))
            
        conn.commit()
        
        elapsed = time.time() - start
        print(f"    -> Discovered {new_people_count} new unique people in {elapsed:.1f}s.")
        print(f"    -> {len(noise_idx)} faces marked as Unknown (Orphans).")

    conn.close()
    print("\n  Cluster calculation complete!")

if __name__ == "__main__":
    main()