"""The single generic runner.

`run(run_cfg)` builds the gate-validated frozen data, the model+adapter, trains, evaluates,
and writes the standardized results directory + run-ledger row. It never branches on the
baking variant beyond the registry lookups and ONE guard (objective.sampler == data sampler).
`main()` is the CLI entry point (`run.py` and the `bakery` console script both call it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from bakery import results as results_mod
from bakery.config import cli_list_to_overrides, config_to_dict, expand_sweep, load_config
from bakery.eval.registry import EvalContext, run_metrics
from bakery.models.peft_factory import build_bundle
from bakery.objectives.base import TeacherLogitCache, collate_framings, get_objective
from bakery.registry import get_experiment, list_experiments
from bakery.seeding import seed_everything
from bakery.trajectories.base import build_dataset


def _ensure_registered():
    import bakery.eval.metrics   # noqa: F401  (register metrics)
    import bakery.experiments    # noqa: F401  (register experiments + their builders)
    import bakery.objectives     # noqa: F401  (register objectives)


def get_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--model.device cuda was requested, but CUDA is not available.")
    return requested


def _log(msg, log_path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(log_path, "a") as f:
        f.write(line + "\n")
    print(line)


def _trainable(bundle):
    return [p for p in bundle.peft_model.parameters() if p.requires_grad]


def _run_epoch(objective, bundle, trajs, optimizer, run_cfg, pad_id, rng, teacher_cache=None,
               scheduler=None):
    bundle.peft_model.train()
    order = list(range(len(trajs)))
    rng.shuffle(order)
    bs = max(int(run_cfg.train.batch_size), 1)
    accum = max(int(run_cfg.train.grad_accum), 1)

    losses = []
    optimizer.zero_grad()
    starts = list(range(0, len(order), bs))
    micro = 0
    for si, start in enumerate(starts):
        idx = order[start: start + bs]
        # traj_keys = the trajectory's stable index in `trajs` (fixed list); used ONLY by the opt-in
        # teacher cache. Ignored downstream when teacher_cache is None → default path byte-identical.
        batch = collate_framings([trajs[i] for i in idx], pad_id, traj_keys=tuple(idx))
        loss = objective.compute_loss(bundle=bundle, batch=batch, cfg=run_cfg, teacher_cache=teacher_cache)
        (loss / accum).backward()
        losses.append(float(loss.item()))
        micro += 1
        if micro % accum == 0 or si == len(starts) - 1:
            if run_cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(_trainable(bundle), run_cfg.train.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
    return sum(losses) / max(len(losses), 1)


def _eval_and_log(epoch, train_kl, bundle, data, run_cfg, spec, history, run_dir, device, log_path):
    names = list(dict.fromkeys(list(run_cfg.eval.metrics) + list(spec.extra_metrics)))
    ctx = EvalContext(bundle=bundle, data=data, run_cfg=run_cfg, device=device)
    results = run_metrics(names, ctx)

    def push(k, v):
        history.setdefault(k, []).append(v)

    push("epochs", epoch)
    push("train_kl", train_kl)
    for k, v in results.items():
        push(k, v)
    results_mod.write_metrics(run_dir, history)
    _log(f"epoch {epoch:>4} | train_kl {train_kl:.6f} | eval_kl {results.get('eval_kl')}", log_path)


def run(run_cfg, sweep_id=None) -> Path:
    _ensure_registered()
    device = get_device(run_cfg.model.device)
    spec = get_experiment(run_cfg.experiment)
    objective = get_objective(run_cfg.train.objective)

    data_seed = run_cfg.seed
    gen_seed = run_cfg.seed if run_cfg.gen_seed is None else run_cfg.gen_seed
    model_seed = run_cfg.seed if run_cfg.model_seed is None else run_cfg.model_seed

    bundle = build_bundle(run_cfg.model)
    data = build_dataset(
        spec.builder_name, run_cfg, bundle=bundle,
        data_seed=data_seed, gen_seed=gen_seed, model_seed=model_seed,
    )

    # The ONLY variant-aware line: trajectories must have been sampled the way the
    # objective expects (e.g. pursuit teaches from the adapter-on distribution).
    if data.spec.sampler != objective.sampler:
        raise ValueError(
            f"Objective {objective.name!r} needs sampler {objective.sampler!r} but trajectories "
            f"were sampled with {data.spec.sampler!r}."
        )

    seed_everything(model_seed)
    run_id = results_mod.make_run_id(run_cfg)
    run_dir = results_mod.prepare_run_dir(run_cfg, run_id)
    log_path = run_dir / "log.txt"
    _log(f"Experiment: {run_cfg.experiment} | objective: {objective.name} | builder: {data.builder_name}", log_path)
    _log(f"Data stats: {data.stats}", log_path)
    results_mod.save_data_artifacts(run_dir, data)

    metrics_schema = ["epochs", "train_kl"] + list(
        dict.fromkeys(list(run_cfg.eval.metrics) + list(spec.extra_metrics))
    )
    results_mod.write_manifest(
        run_dir, run_cfg=run_cfg, run_id=run_id, data=data,
        num_trainable_params=bundle.num_trainable_params(), status="running",
        metrics_schema=metrics_schema, sweep_id=sweep_id,
    )
    config_digest = hashlib.sha1(
        json.dumps(config_to_dict(run_cfg), sort_keys=True).encode()
    ).hexdigest()[:12]
    results_mod.append_run_log({
        "run_id": run_id, "experiment": run_cfg.experiment, "objective": objective.name,
        "sweep_id": sweep_id, "backend": run_cfg.backend, "host": platform.node(),
        "device": device, "model": run_cfg.model.name,
        "base_prompt": run_cfg.generation.base_prompt, "baked_prompt": run_cfg.generation.baked_prompt,
        "config_digest": config_digest, "data_stats": data.stats, "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
    }, output_root=run_cfg.output_root)

    if run_cfg.train.optimizer != "AdamW":
        raise ValueError(f"Unsupported optimizer: {run_cfg.train.optimizer}")
    optimizer = torch.optim.AdamW(
        _trainable(bundle), lr=run_cfg.train.learning_rate, weight_decay=run_cfg.train.weight_decay
    )

    # Optional LR schedule (convergence-speed lever). "constant" → scheduler=None → fixed LR (unchanged).
    scheduler = None
    sched = getattr(run_cfg.train, "lr_schedule", "constant")
    if sched and sched != "constant":
        bs_ = max(int(run_cfg.train.batch_size), 1)
        accum_ = max(int(run_cfg.train.grad_accum), 1)
        n_batches = -(-len(data.train_trajectories) // bs_)        # ceil
        steps_per_epoch = -(-n_batches // accum_)                  # ceil (one opt-step per accum boundary)
        total_steps = max(steps_per_epoch * int(run_cfg.train.num_epochs), 1)
        warmup_steps = int(max(0.0, float(getattr(run_cfg.train, "warmup_frac", 0.0))) * total_steps)
        from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
        builder = {"cosine": get_cosine_schedule_with_warmup,
                   "linear": get_linear_schedule_with_warmup}.get(sched)
        if builder is None:
            raise ValueError(f"Unknown train.lr_schedule {sched!r}; use constant|cosine|linear.")
        scheduler = builder(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    pad_id = data.tokenizer_fingerprint.pad_id
    rng = random.Random(model_seed)
    history: dict = {}
    ckpt_dir = run_dir / "checkpoints"

    # Opt-in teacher-logit cache (default "off" → teacher_cache stays None → byte-identical path).
    # Refused for objectives whose teacher changes each epoch (pursue): caching a moving target is wrong.
    teacher_cache = None
    ctl = getattr(run_cfg.train, "cache_teacher_logits", "off")
    if ctl and ctl != "off":
        if objective.needs_per_epoch_trajectories:
            raise ValueError(
                f"train.cache_teacher_logits={ctl!r} is incompatible with objective {objective.name!r}: "
                f"its teacher changes each epoch (per-epoch trajectories), so it is not memoizable."
            )
        teacher_cache = TeacherLogitCache(
            backend=ctl,
            verify_every=getattr(run_cfg.train, "cache_teacher_verify_every", 0),
            dtype=getattr(run_cfg.train, "cache_teacher_dtype", "float32"),
            verify_atol=getattr(run_cfg.train, "cache_teacher_verify_atol", 1e-2),
        )

    try:
        for epoch in range(1, run_cfg.train.num_epochs + 1):
            if objective.needs_per_epoch_trajectories and epoch > 1:
                data = build_dataset(
                    spec.builder_name, run_cfg, bundle=bundle,
                    data_seed=data_seed, gen_seed=gen_seed + epoch, model_seed=model_seed,
                )
            train_kl = _run_epoch(objective, bundle, list(data.train_trajectories),
                                  optimizer, run_cfg, pad_id, rng, teacher_cache=teacher_cache,
                                  scheduler=scheduler)
            if epoch % run_cfg.train.eval_period == 0:
                _eval_and_log(epoch, train_kl, bundle, data, run_cfg, spec, history,
                              run_dir, device, log_path)
            if run_cfg.train.save_every and epoch % run_cfg.train.save_every == 0 \
                    and run_cfg.train.save_adapters:
                bundle.save_adapter(ckpt_dir / f"epoch_{epoch}")
    except BaseException:
        results_mod.update_manifest_status(run_dir, "failed")
        results_mod.update_run_log(
            run_id, output_root=run_cfg.output_root, status="failed",
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        raise

    if run_cfg.train.save_adapters:
        bundle.save_adapter(ckpt_dir / "final")
    results_mod.update_manifest_status(run_dir, "completed")
    results_mod.update_run_log(
        run_id, output_root=run_cfg.output_root, status="completed",
        finished_utc=datetime.now(timezone.utc).isoformat(),
        **results_mod.headline_from_metrics(history),
    )
    _log(f"DONE. results in {run_dir}", log_path)
    return run_dir


# ======================================================================================
# CLI
# ======================================================================================

def main(argv=None) -> None:
    _ensure_registered()
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        description="Run a Bakery experiment. Unknown --dotted.key value pairs override config "
                    "fields (e.g. --model.lora_rank 16 --generation.num_contexts 200)."
    )
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="YAML override file")
    parser.add_argument("--sweep", type=str, default=None, help="YAML sweep spec")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--list", action="store_true", help="List registered experiments and exit")
    known, unknown = parser.parse_known_args(argv)

    if known.list:
        print("Registered experiments:")
        for name in list_experiments():
            print(f"  {name}: {get_experiment(name).description}")
        return

    if known.sweep:
        import yaml
        with open(known.sweep) as f:
            sweep = yaml.safe_load(f)
        sweep_id = uuid.uuid4().hex[:8]
        overrides_cli = cli_list_to_overrides(unknown)
        points = expand_sweep(sweep)
        print(f"Sweep {sweep_id}: {len(points)} run(s)")
        for experiment, point_overrides in points:
            merged = {**point_overrides, **overrides_cli}
            run(load_config(experiment, cli_overrides=merged), sweep_id=sweep_id)
        return

    if not known.experiment:
        parser.error("--experiment is required (or use --sweep / --list).")

    overrides = cli_list_to_overrides(unknown)
    run_cfg = load_config(known.experiment, cli_overrides=overrides, yaml_path=known.config)

    if known.print_config:
        print(json.dumps(config_to_dict(run_cfg), indent=2))
        return

    run(run_cfg)


if __name__ == "__main__":
    main()
