from __future__ import annotations

import re

from moose_api.main import _new_job_id


def test_new_job_id_is_dashless_uuid_hex() -> None:
    job_id = _new_job_id()
    assert "-" not in job_id
    assert len(job_id) == 32
    assert re.fullmatch(r"[0-9a-f]{32}", job_id) is not None
