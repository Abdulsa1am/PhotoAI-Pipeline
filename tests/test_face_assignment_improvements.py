import importlib.util
import os
import unittest

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_script_module(module_name, script_filename):
    script_path = os.path.join(REPO_ROOT, script_filename)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


if __name__ == "__main__":
    unittest.main()
