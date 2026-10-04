import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd, numpy as np
from s09b_kdd_shared_features import shared_features
def test_synthetic_enrolment():
    dates = pd.DataFrame({"course_id": ["c"], "from": ["2014-01-01"], "to": ["2014-01-30"]})
    enrol = pd.DataFrame({"enrollment_id": [1, 2], "course_id": ["c", "c"]})
    log = pd.DataFrame({"enrollment_id": [1, 1, 1, 1, 1], "time": ["2014-01-01T10:00:00", "2014-01-01T11:00:00", "2014-01-16T09:00:00", "2014-01-28T09:00:00", "2014-01-29T09:00:00"],
                        "event": ["problem", "discussion", "video", "problem", "problem"]})
    f = shared_features(log, enrol, dates).set_index("enrollment_id")
    a = f.loc[1]; b = f.loc[2]
    assert a.f01 == 4 and a.f02 == 3 and a.f05 == 0 and a.f04 == 0          # day-28 event (Jan 29) excluded; last day 27
    assert a.f08 == 1 and a.f09 == 2 and np.isclose(a.f06, (2 + 1) / (2 + 1))
    assert b.f01 == 0 and b.f04 == 28 and b.f05 == 28 and b.f06 == 1 and b.f07 == 0
