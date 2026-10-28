import os

from hypothesis import HealthCheck, settings

# CI детерминирован: фиксированный seed генератора и конечное число примеров (ARCHITECTURE §11).
settings.register_profile(
    "ci",
    derandomize=True,
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", max_examples=100, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
