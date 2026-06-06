"""The always-destroy guarantee: a remote run tears the box down even when the run errors.

Pure-Python (monkeypatched subprocess boundary) — never touches a real vast account.
"""

import pytest

import vast.remote as r


def test_remote_run_always_destroys(monkeypatch):
    destroyed = []
    monkeypatch.setattr(r, "find_cheapest_offer", lambda mp, gpu=None: {"id": 1, "dph_total": 0.2})
    monkeypatch.setattr(r, "create_instance", lambda offer, mp: "INST1")
    monkeypatch.setattr(r, "_ssh_endpoint", lambda iid, **k: ("host", 22))
    monkeypatch.setattr(r, "_sync_up", lambda host, port: None)
    monkeypatch.setattr(r, "_sync_down", lambda host, port: None)
    monkeypatch.setattr(r, "destroy_instance", lambda iid: destroyed.append(iid))

    def boom(host, port, run_args):
        raise RuntimeError("run failed on the box")

    monkeypatch.setattr(r, "_exec", boom)

    with pytest.raises(RuntimeError):
        r.remote_run(["--experiment", "bake_smoke"], max_price=0.60)

    assert destroyed == ["INST1"], "the box must be destroyed even when the run errors"
