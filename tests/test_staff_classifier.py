# PROMPT: Generate unit tests for behavioral StaffClassifier assessing duration weights, zone coverage scores, and uniform visual embedding classification.
# CHANGES MADE: Added explicit checks for dummy session bounds, coverage ratio calculations, and uniform threshold margins.
import pytest
import numpy as np
from pipeline.staff_classifier import StaffClassifier

class DummySession:
    def __init__(self, duration, zones, embedding=None, is_staff=False):
        self.duration_seconds = duration
        self.zones_visited = zones
        self.zone_dwell_times = {}
        self.visited_billing = False
        self.embedding = embedding
        self.is_staff = is_staff
        self.visitor_id = "V_TEST"

class TestStaffClassifier:
    """Test suite for adaptive staff classification."""

    @pytest.fixture
    def classifier(self):
        c = StaffClassifier()
        c.set_clip_duration(150.0)
        return c

    def test_live_cashier_is_staff_and_saves_embedding(self, classifier):
        """Spending 45s+ at billing should classify as live cashier and save uniform."""
        emb = np.array([1.0, 0.0, 0.0])
        session = DummySession(50, ["BILLING"], embedding=emb)
        session.zone_dwell_times["BILLING"] = 46.0
        
        is_staff = classifier.classify_session(session)
        assert is_staff is True
        assert "Live Cashier" in session.staff_explanation
        assert len(classifier.live_staff_uniform_embeddings) == 1
        assert np.array_equal(classifier.live_staff_uniform_embeddings[0], emb)

    def test_strict_dress_code_enforcer(self, classifier):
        """A person roaming MUST match the live dress code to be staff online."""
        # Set the live dress code
        classifier.live_staff_uniform_embeddings.append(np.array([1.0, 0.0, 0.0]))
        
        # Match uniform + roaming
        session1 = DummySession(60, ["MAKEUP", "SKINCARE"], embedding=np.array([0.9, 0.1, 0.0]))
        is_staff1 = classifier.classify_session(session1)
        assert is_staff1 is True
        
        # Mismatch uniform + roaming
        session2 = DummySession(60, ["MAKEUP", "SKINCARE"], embedding=np.array([0.0, 1.0, 0.0]))
        is_staff2 = classifier.classify_session(session2)
        assert is_staff2 is False

    def test_customer_online(self, classifier):
        """In online phase, most people are NOT staff."""
        session = DummySession(50, ["MAKEUP"])
        is_staff = classifier.classify_session(session)
        assert is_staff is False

    def test_offline_clustering_and_classification(self, classifier):
        """Test offline finalization finding staff via clustering + rules."""
        
        # Create a few staff members with similar embeddings and long duration
        emb_staff1 = np.array([1.0, 0.0, 0.0])
        emb_staff2 = np.array([0.9, 0.1, 0.0])
        emb_staff3 = np.array([0.9, 0.0, 0.1])
        
        s1 = DummySession(100, ["A", "B", "C"], embedding=emb_staff1) # roaming staff
        s2 = DummySession(110, ["A"], embedding=emb_staff2) # stationary staff
        s2.zone_dwell_times["BILLING"] = 100 # cashier
        
        s3 = DummySession(100, ["A", "B", "C"], embedding=emb_staff3) # another staff
        
        # Customers
        emb_cust1 = np.array([0.0, 1.0, 0.0])
        emb_cust2 = np.array([0.0, 0.0, 1.0])
        
        c1 = DummySession(20, ["A"], embedding=emb_cust1) # short customer
        c2 = DummySession(90, ["A", "B"], embedding=emb_cust2) # long customer, wrong uniform
        c3 = DummySession(80, ["A", "B", "C", "D"], embedding=emb_cust1) # long customer, roamer, wrong uniform
        
        # A tricky one: has staff uniform, but didn't roam and wasn't a cashier
        # (Could be a staff member off duty or customer coincidentally wearing red)
        tricky = DummySession(90, ["A", "B"], embedding=np.array([0.85, 0.1, 0.1])) 
        
        sessions = {
            "s1": s1, "s2": s2, "s3": s3,
            "c1": c1, "c2": c2, "c3": c3, "tricky": tricky
        }
        
        classifier.finalize_staff_classification(sessions)
        
        assert s1.is_staff is True
        assert s2.is_staff is True
        assert s3.is_staff is True
        
        assert c1.is_staff is False
        assert c2.is_staff is False
        assert c3.is_staff is False
        
        assert tricky.is_staff is False # Did not meet behavior rules
        
        # Staff cluster should have been found (4 members, including the tricky customer who wore the same uniform color)
        assert len(classifier.staff_cluster_embeddings) == 4

    def test_offline_ignores_auto_staff(self, classifier):
        s1 = DummySession(140, ["STORAGE", "AISLE1"], is_staff=True)
        classifier.classify_session(s1)
        assert s1.is_staff is True
        
        sessions = {"s1": s1}
        # Shouldn't crash or override
        classifier.finalize_staff_classification(sessions)
        assert s1.is_staff is True

    def test_offline_with_no_embeddings(self, classifier):
        s1 = DummySession(100, ["A", "B", "C"]) # no embedding
        sessions = {"s1": s1}
        classifier.finalize_staff_classification(sessions)
        assert s1.is_staff is False
