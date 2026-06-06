"""The always-destroy guarantee: a remote run tears the box down even when the run errors.

Pure-Python (monkeypatched at the vast-CLI / SSH boundary) — never touches a real vast account.
"""

import pytest

import vast.remote as r


def test_remote_run_always_destroys(monkeypatch):
    destroyed = []
    monkeypatch.setattr(r, "ensure_account_key", lambda: None)
    monkeypatch.setattr(r, "find_offer", lambda mp: {"id": 1, "gpu_name": "A40", "gpu_ram": 46068, "dph_total": 0.3})
    monkeypatch.setattr(r, "create_instance", lambda offer: "INST1")
    monkeypatch.setattr(r, "_attach_key", lambda iid: None)
    monkeypatch.setattr(r, "_ssh_endpoint", lambda iid, **k: ("host", 22))
    monkeypatch.setattr(r, "_upload", lambda host, port: None)
    monkeypatch.setattr(r, "_write_hf_token", lambda host, port: None)
    monkeypatch.setattr(r, "_download", lambda host, port: None)
    monkeypatch.setattr(r, "destroy_instance", lambda iid: destroyed.append(iid))

    def boom(host, port, run_args):
        raise RuntimeError("run failed on the box")

    monkeypatch.setattr(r, "_exec", boom)

    with pytest.raises(RuntimeError):
        r.remote_run(["--experiment", "bake_squad"], max_price=0.80)

    assert destroyed == ["INST1"], "the box must be destroyed even when the run errors"
