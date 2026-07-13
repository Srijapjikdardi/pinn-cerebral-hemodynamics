"""
Phase 3 — Training.

Two-stage optimization, as specified in the plan:
  1) Adam for `training.adam_epochs` epochs — good at making fast, robust
     early progress on the highly non-convex multi-term PINN loss landscape.
  2) L-BFGS for fine-tuning — second-order method that converges to a much
     sharper optimum once Adam has found a good basin, standard practice for
     PINNs (Raissi et al., 2019).

Self-adaptive loss weighting (models/losses.SelfAdaptiveWeights) is jointly
optimized alongside the network weights so the mass/momentum/boundary/junction
terms don't have to be hand-balanced.

Run:
    python -m training.train --config configs/default.yaml --data data/synthetic_dataset.h5
"""

import argparse
import os
import yaml
import h5py
import numpy as np
import torch
from tqdm import tqdm

from data.network_topology import SEGMENTS, junction_map, SEGMENT_LENGTHS_M
from models.gpinn import build_model
from models.losses import pde_residuals, data_loss, boundary_loss, junction_loss, SelfAdaptiveWeights

SEGMENT_NAMES = [s[0] for s in SEGMENTS]
SEG_ID_LOOKUP = {name: i for i, name in enumerate(SEGMENT_NAMES)}


def load_dataset(h5_path, exclude_last_n=0):
    """Loads the HDF5 dataset produced by data/generate_synthetic.py into memory.

    exclude_last_n: drop the last N patients (by sorted key) from what's returned.
    MUST match validation/evaluate.py's holdout convention (sorted keys, last
    n_validation_holdout patients) — otherwise the model trains on the same patients
    evaluate.py calls "held out," making the MAPE number meaningless. This parameter
    exists specifically to prevent that; train() below always passes it.

    For datasets too large for RAM, replace this with a lazy h5py-backed Dataset
    + DataLoader — the training loop below is agnostic to how batches are sourced."""
    patients = []
    with h5py.File(h5_path, "r") as h5:
        keys = sorted(h5.keys())
        if exclude_last_n > 0:
            keys = keys[:-exclude_last_n]
        for key in keys:
            grp = h5[key]
            cond = grp["conditioning"]["vector"][:]
            segs = {}
            for sk in grp.keys():
                if sk.startswith("segment_"):
                    seg_name = sk[len("segment_"):]
                    segs[seg_name] = {k: grp[sk][k][:] for k in grp[sk].keys()}
            patients.append({"cond": cond, "segments": segs, "E": grp.attrs["stiffness_E_pa"]})
    return patients


def sample_collocation_batch(patients, batch_size, device):
    """Randomly sample (x, t, seg_id, cond) collocation points across patients/segments
    for physics-residual evaluation (no ground truth needed here)."""
    xs, ts, seg_ids, conds = [], [], [], []
    for _ in range(batch_size):
        p = patients[np.random.randint(len(patients))]
        seg_name = np.random.choice(list(p["segments"].keys()))
        xs.append(np.random.uniform(0, 1))
        ts.append(np.random.uniform(0, 1))
        seg_ids.append(SEG_ID_LOOKUP[seg_name])
        conds.append(p["cond"])
    x = torch.tensor(xs, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    t = torch.tensor(ts, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    seg_id = torch.tensor(seg_ids, dtype=torch.long, device=device)
    cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=device)
    return x, t, seg_id, cond


def sample_data_batch(patients, batch_size, device):
    """Sample (x, t, seg_id, cond) -> (A, Q, P) supervised pairs from the synthetic ground truth."""
    xs, ts, seg_ids, conds, As, Qs, Ps = [], [], [], [], [], [], []
    for _ in range(batch_size):
        p = patients[np.random.randint(len(patients))]
        seg_name = np.random.choice(list(p["segments"].keys()))
        seg = p["segments"][seg_name]
        ix = np.random.randint(seg["A"].shape[1])
        it = np.random.randint(seg["A"].shape[0])
        xs.append(seg["x"][ix] / max(seg["x"][-1], 1e-8))
        ts.append(it / max(seg["A"].shape[0] - 1, 1))
        seg_ids.append(SEG_ID_LOOKUP[seg_name])
        conds.append(p["cond"])
        As.append(seg["A"][it, ix]); Qs.append(seg["Q"][it, ix]); Ps.append(seg["P"][it, ix])

    def col(v):
        return torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(1)

    x, t = col(xs), col(ts)
    seg_id = torch.tensor(seg_ids, dtype=torch.long, device=device)
    cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=device)
    return x, t, seg_id, cond, col(As), col(Qs), col(Ps)


def sample_boundary_batch(patients, batch_size, sweep_cfg, device):
    """Sample true inlet boundary points: x=0 on ICA_L/ICA_R (the network's actual
    inlets — every patient has both, they're never optional, see
    data/network_topology.py:SEGMENTS), with the target Q taken from the same
    physiological waveform data/generate_synthetic.py's solver was driven with,
    split evenly across the two ICAs exactly as the solver does it.

    This replaces an earlier version of this loss that used an all-zero target
    applied to arbitrary (non-inlet) collocation points — that was actively teaching
    the network the wrong physics (zero flow everywhere at x=0), not just an inactive
    placeholder."""
    from data.generate_synthetic import physiological_ica_waveform

    xs, ts, seg_ids, conds, q_targets = [], [], [], [], []
    cycle_s = sweep_cfg["cardiac_cycle_s"]
    for _ in range(batch_size):
        p = patients[np.random.randint(len(patients))]
        seg_name = np.random.choice(["ICA_L", "ICA_R"])
        t_norm = np.random.uniform(0, 1)
        t_phys = t_norm * cycle_s
        q_in = physiological_ica_waveform(np.array([t_phys]), cycle_s)[0] * 1e-6 / 2.0  # mL/s -> m^3/s, split across 2 ICAs
        xs.append(0.0)
        ts.append(t_norm)
        seg_ids.append(SEG_ID_LOOKUP[seg_name])
        conds.append(p["cond"])
        q_targets.append(q_in)

    def col(v):
        return torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(1)

    x, t = col(xs), col(ts)
    seg_id = torch.tensor(seg_ids, dtype=torch.long, device=device)
    cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=device)
    return x, t, seg_id, cond, col(q_targets)


def sample_junction_batches(patients, batch_size_per_node, device):
    """Build one batch per real (multi-segment) junction node for the Kirchhoff loss.

    Limitation, documented not hidden: only patients where EVERY segment touching a
    given node is present are included for that node's batch (e.g. a patient missing
    PCoA_L is excluded from ACA_A1_L_jn and PCA_P1_L_jn batches). This under-trains the
    junction constraint for incomplete-circle patients specifically at their missing
    junctions — acceptable for now since every node still has 12-18/20 demo patients
    valid, but worth revisiting (e.g. a variable-arity junction formulation) if a real
    sweep shows heavy anatomical variation clusters.

    Note on PCA_P1_L_jn / PCA_P1_R_jn: both segments touching these nodes have sign +1
    in junction_map() (both "arrive"), since the distal P2/PCA branch beyond this point
    isn't modeled as its own segment. Enforcing sum(sign*Q)=0 there is therefore not
    literal Kirchhoff (there IS a real distal outflow in reality) — but it exactly
    matches the convention data/generate_synthetic.py's solver already used to produce
    the Phase 1 ground truth, so it's the right target for the G-PINN to match, not a bug.

    Returns a list of (junction_points_list, cond) tuples — each ready to pass straight
    into models.losses.junction_loss(model, junction_points_list, cond, SEG_ID_LOOKUP).
    """
    jmap = junction_map()
    real_nodes = {n: t for n, t in jmap.items() if n != "BA" and len(t) >= 2}

    batches = []
    for node, touching in real_nodes.items():
        valid_patients = [p for p in patients if all(seg in p["segments"] for seg, _ in touching)]
        if not valid_patients:
            continue
        conds, ts = [], []
        for _ in range(batch_size_per_node):
            p = valid_patients[np.random.randint(len(valid_patients))]
            conds.append(p["cond"])
            ts.append(np.random.uniform(0, 1))
        cond = torch.tensor(np.stack(conds), dtype=torch.float32, device=device)
        t = torch.tensor(ts, dtype=torch.float32, device=device).unsqueeze(1)
        B = len(conds)

        xs, seg_ids, signs = [], [], []
        for seg_name, sign in touching:
            x_val = 1.0 if sign == +1 else 0.0  # sign+1 = arrives at node = far (x=1) end of that segment
            xs.append(torch.full((B, 1), x_val, device=device))
            seg_ids.append(torch.full((B,), SEG_ID_LOOKUP[seg_name], dtype=torch.long, device=device))
            signs.append(sign)

        jp_entry = {"segments": [s for s, _ in touching], "signs": signs, "x": xs, "t": t, "seg_ids": seg_ids}
        batches.append(([jp_entry], cond))
    return batches


def build_seg_props(seg_id_batch, cond_batch, solver_cfg, cycle_s, device):
    """Look up per-point (rho, friction_f, E, h, r0, A0, L_m, T_s) needed by the PDE residuals.

    r0/A0 are derived from each point's *actual patient-specific* diameter (read out of
    its conditioning vector), not a generic baseline table — this must match
    models/gpinn.py's GPINN._local_area_scale exactly, or the physics residual's tube-law
    reference area disagrees with the one the model itself uses internally.

    L_m/T_s are the chain-rule correction factors for the fact that collocation x, t are
    sampled normalized to [0,1], not physical meters/seconds: d/dx_norm = L_m * d/dx_phys,
    so d/dx_phys = (1/L_m) * d/dx_norm. Without this, PDE residuals are scaled
    inconsistently across segments of different length (shorter segments get artificially
    larger residuals for the same physical error) — doesn't change where the residual is
    zero, but biases the optimizer's attention. See models/losses.py:pde_residuals.
    """
    B = seg_id_batch.shape[0]
    rho_val = solver_cfg["blood_density_kg_m3"]
    nu_val = solver_cfg["blood_dynamic_viscosity_pa_s"] / rho_val
    rho = torch.full((B, 1), rho_val, device=device)
    # friction_f folds in kinematic viscosity so that models/losses.py's
    # `friction_f * Q / A` term matches data/generate_synthetic.py's f*nu*Q/A exactly.
    friction_f = torch.full((B, 1), solver_cfg["friction_coefficient"] * nu_val, device=device)
    # E stored as the last element of the conditioning vector (see PatientTopology.conditioning_vector)
    E = cond_batch[:, -1:] * 2.0e6  # undo the normalization applied at dataset build time
    # Diameters stored as cond[:, :n_segments], normalized by /5.0 mm (see
    # PatientTopology.conditioning_vector) — gather each point's own segment's value,
    # exactly as models/gpinn.py:GPINN._local_area_scale does.
    diam_norm = cond_batch.gather(1, seg_id_batch.unsqueeze(1))
    diam_mm = diam_norm * 5.0
    r0 = (diam_mm / 1000.0) / 2.0
    A0 = np.pi * r0 ** 2
    h = solver_cfg["wall_thickness_ratio"] * r0

    length_table = torch.tensor(SEGMENT_LENGTHS_M, dtype=torch.float32, device=device)
    L_m = length_table[seg_id_batch].unsqueeze(1)
    T_s = torch.full((B, 1), float(cycle_s), device=device)

    return {"rho": rho, "friction_f": friction_f, "E": E, "h": h, "r0": r0, "A0": A0,
            "L_m": L_m, "T_s": T_s}


def train(config_path, data_path, out_dir="outputs", resume_from=None, reset_schedule=False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(out_dir, exist_ok=True)

    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["topology"], cfg["model"]).to(device)

    term_names = ["data", "mass", "momentum", "boundary", "junction"]
    adaptive = SelfAdaptiveWeights(term_names).to(device)
    fixed_weights = cfg["loss_weights"]

    patients = load_dataset(data_path, exclude_last_n=cfg["sweep"]["n_validation_holdout"])
    params = list(model.parameters()) + (list(adaptive.parameters()) if fixed_weights["self_adaptive"] else [])
    optimizer = torch.optim.Adam(params, lr=cfg["training"]["adam_lr"])
    # Cosine decay: constant adam_lr for thousands of epochs was confirmed (empirically,
    # across several resumed runs) to cause oscillation around the loss landscape rather
    # than settling into it. Decaying to ~1% of the initial LR by the end of Adam fixes this
    # without needing per-run manual tuning.
    #
    # T_max must be the TOTAL cumulative epoch budget across the whole training run (all
    # resumes combined), not just this call's chunk — otherwise each resume recreates the
    # scheduler with T_max = this-call's adam_epochs, but the internal step counter (restored
    # from the checkpoint) keeps advancing past that T_max on every subsequent resume. Past
    # the trough, cosine's periodicity means LR climbs back UP instead of staying decayed —
    # confirmed directly: LR climbed back to ~0.0009 (near the original 0.001) after a chain
    # of resumes, and training regressed as a result. `total_adam_epochs` defaults to
    # `adam_epochs` for single-call runs (no behavior change there); set it explicitly and
    # higher than `adam_epochs` when you know you'll be chaining resumed calls.
    total_epochs = cfg["training"].get("total_adam_epochs", cfg["training"]["adam_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=cfg["training"]["adam_lr"] * 0.01)

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        # Checkpoints are dicts with "model"/"optimizer" keys going forward; older
        # checkpoints (pre-fix) were a raw model state_dict — handle both.
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if reset_schedule:
                # Intentional fine-tune with a NEW adam_lr/total_adam_epochs: do NOT restore
                # optimizer/scheduler state. Found by direct bug: restoring scheduler.state_dict()
                # silently overwrites the freshly-constructed scheduler's base_lrs and step count
                # with the SOURCE run's values — a fine-tune requesting a lower peak LR (e.g. 1e-5)
                # would get its schedule silently hijacked by the old run's base_lr (e.g. 1e-3) and
                # step count, producing a nonsensical lr (confirmed: got 0.000142 at the end of a
                # supposed-peak-1e-5 fine-tune, 14x higher than the peak itself). Model weights
                # still transfer; only the LR schedule and Adam moments start fresh.
                print(f"Resumed weights only from {resume_from} (--reset_schedule: fresh "
                      f"optimizer/scheduler, using this run's own adam_lr/total_adam_epochs)")
            elif "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
                print(f"Resumed weights + optimizer state from {resume_from}")
                if "scheduler" in ckpt:
                    ckpt_sched = ckpt["scheduler"]
                    ckpt_T_max = ckpt_sched.get("T_max")
                    ckpt_base_lrs = ckpt_sched.get("base_lrs")
                    mismatch = (ckpt_T_max is not None and ckpt_T_max != total_epochs) or \
                               (ckpt_base_lrs is not None and
                                any(abs(b - cfg["training"]["adam_lr"]) > 1e-12 for b in ckpt_base_lrs))
                    if mismatch:
                        print(f"WARNING: checkpoint's scheduler state (T_max={ckpt_T_max}, "
                              f"base_lrs={ckpt_base_lrs}) does not match this run's config "
                              f"(total_adam_epochs={total_epochs}, adam_lr={cfg['training']['adam_lr']}). "
                              f"Restoring it anyway will silently override this run's intended LR/schedule "
                              f"with the checkpoint's — this is the exact bug class that produced a wrong "
                              f"lr=0.000142 for an intended-1e-5-peak fine-tune earlier in this project. "
                              f"If you meant to continue the SAME schedule, your config should match the "
                              f"checkpoint's — if you meant to fine-tune with a NEW schedule, pass "
                              f"--reset_schedule instead.")
                    scheduler.load_state_dict(ckpt_sched)
            else:
                print(f"Resumed weights from {resume_from} (no optimizer state in checkpoint — "
                      f"fresh Adam moments, expect a transient lurch on continuation)")
        else:
            model.load_state_dict(ckpt)
            print(f"Resumed weights from {resume_from} (old-format checkpoint, no optimizer "
                  f"state available — fresh Adam moments, expect a transient lurch on continuation)")

    def compute_total_loss():
        # data term
        x_d, t_d, seg_d, cond_d, A_t, Q_t, P_t = sample_data_batch(
            patients, cfg["training"]["batch_size_data"], device)
        norm_scales = {"A_scale": 1.0e-5, "Q_scale": 1.5e-6, "P_scale": 5000.0}  # P_scale here is
        # only for normalizing the supervised data_loss comparison — P itself is no longer a
        # learned/scaled network output (see models/gpinn.py forward() docstring), this is
        # just "how many Pa counts as a unit of error" for the loss.
        L_data = data_loss(model, x_d, t_d, seg_d, cond_d, A_t, Q_t, P_t, norm_scales)

        # physics terms
        x_c, t_c, seg_c, cond_c = sample_collocation_batch(
            patients, cfg["training"]["batch_size_collocation"], device)
        seg_props = build_seg_props(seg_c, cond_c, cfg["solver"], cfg["sweep"]["cardiac_cycle_s"], device)
        res = pde_residuals(model, x_c, t_c, seg_c, cond_c, seg_props)
        L_mass = (res["mass"] ** 2).mean()
        L_momentum = (res["momentum"] ** 2).mean() + (res["tube_law"] ** 2).mean()

        # boundary term: true ICA inlet flow, matching the physiological waveform the
        # Phase 1 solver was actually driven with (see sample_boundary_batch docstring
        # for why this replaced an earlier, physically-wrong all-zero-target version)
        x_bc, t_bc, seg_bc, cond_bc, Q_bc_target = sample_boundary_batch(
            patients, cfg["training"]["batch_size_collocation"] // 4, cfg["sweep"], device)
        L_boundary = boundary_loss(model, x_bc, t_bc, seg_bc, cond_bc, Q_bc_target)

        # junction term — real Kirchhoff/pressure-continuity constraint at each
        # multi-segment branch node, per-node-type batches (see sample_junction_batches
        # docstring for the terminal-node convention note and the per-node validity
        # limitation for anatomically-incomplete patients).
        junction_batches = sample_junction_batches(patients, batch_size_per_node=32, device=device)
        if junction_batches:
            L_junction = sum(
                junction_loss(model, jp_list, cond, SEG_ID_LOOKUP) for jp_list, cond in junction_batches
            ) / len(junction_batches)
        else:
            L_junction = torch.tensor(0.0, device=device)

        losses = {"data": L_data, "mass": L_mass, "momentum": L_momentum,
                  "boundary": L_boundary, "junction": L_junction}

        if fixed_weights["self_adaptive"]:
            total, breakdown = adaptive(losses)
        else:
            total = sum(fixed_weights[k] * losses[k] for k in term_names)
            breakdown = {k: v.item() for k, v in losses.items()}
        return total, breakdown

    # ---- Stage 1: Adam ----
    pbar = tqdm(range(cfg["training"]["adam_epochs"]), desc="Adam")
    postfix_every = max(1, min(500, cfg["training"]["adam_epochs"] // 20))
    history = []  # (epoch, lr, data, mass, momentum, boundary, junction) — saved to CSV below,
                  # for plotting a real convergence curve rather than only spot-checking loss values
    for epoch in pbar:
        optimizer.zero_grad()
        total, breakdown = compute_total_loss()
        total.backward()
        optimizer.step()
        scheduler.step()
        history.append((epoch, scheduler.get_last_lr()[0], breakdown.get("data", 0.0),
                         breakdown.get("mass", 0.0), breakdown.get("momentum", 0.0),
                         breakdown.get("boundary", 0.0), breakdown.get("junction", 0.0)))
        if epoch % postfix_every == 0:
            breakdown["lr"] = scheduler.get_last_lr()[0]
            pbar.set_postfix(breakdown)
        if epoch % cfg["training"]["checkpoint_every"] == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
                       os.path.join(out_dir, f"gpinn_adam_{epoch}.pt"))

    with open(os.path.join(out_dir, "loss_history.csv"), "w") as f:
        f.write("epoch,lr,data,mass,momentum,boundary,junction\n")
        for row in history:
            f.write(",".join(str(v) for v in row) + "\n")

    # Always save full state at the TRUE end of the Adam loop, not just at
    # checkpoint_every boundaries — otherwise a run that completes its full
    # adam_epochs range (exit 0, not a timeout) has no resumable checkpoint at its
    # actual endpoint (the last periodic save could be up to checkpoint_every-1
    # epochs behind), forcing any further continuation to redo epochs it already did.
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()},
               os.path.join(out_dir, "gpinn_adam_end.pt"))

    # ---- Stage 2: L-BFGS fine-tune ----
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=cfg["training"]["lbfgs_max_iter"],
        history_size=cfg["training"]["lbfgs_history_size"],
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        total, _ = compute_total_loss()
        total.backward()
        return total

    lbfgs.step(closure)

    final_path = os.path.join(out_dir, "gpinn_final.pt")
    torch.save({"model": model.state_dict()}, final_path)  # no "optimizer" key: this is
    # post-LBFGS, an Adam optimizer resuming from here starts fresh (expected/fine, LBFGS
    # state isn't meaningful to hand back to Adam anyway)
    print(f"Training complete. Saved final model to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data/synthetic_dataset.h5")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--resume_from", default=None, help="Checkpoint path to resume weights from")
    parser.add_argument("--reset_schedule", action="store_true",
                         help="Load weights only, fresh optimizer/scheduler (for fine-tuning with "
                              "a different adam_lr/total_adam_epochs than the source checkpoint used)")
    args = parser.parse_args()
    train(args.config, args.data, args.out_dir, args.resume_from, args.reset_schedule)
