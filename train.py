import argparse
import os
import pickle
import secrets
import yaml
import random
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import trange

from preprocess import preprocess_trace
from io_utils import REQUIRED_PATH_KEYS
from adjacency import build_graphs_and_edges
from models import TRACE
from losses import vae_loss
from metrics import evaluate_all
from plotting import draw_spatial_pies, resolve_celltype_colors


def seed_everything(seed, deterministic=True):
    seed = int(seed)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(options):
    requested = str(options.get('device', 'auto')).lower()
    if requested == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable")
    return torch.device(requested)


def resolve_run_seed(run_cfg, legacy_split_seed=None):
    mode = str(run_cfg.get('seed_mode', 'fixed')).lower()
    if mode == 'random':
        return secrets.randbelow(2**31 - 1), mode
    if mode != 'fixed':
        raise ValueError("run.seed_mode must be 'fixed' or 'random'")
    configured = run_cfg.get('seed', legacy_split_seed)
    if configured is None:
        configured = 0
    return int(configured), mode


def apply_config_overrides(cfg, overrides):
    """Apply repeatable dotted YAML overrides, primarily for smoke/debug runs."""
    for expression in overrides or []:
        if '=' not in expression:
            raise ValueError(f"Config override must be KEY=VALUE, got {expression!r}")
        dotted_key, raw_value = expression.split('=', 1)
        keys = dotted_key.split('.')
        target = cfg
        for key in keys[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise TypeError(f"Cannot set nested override below {key!r}")
        target[keys[-1]] = yaml.safe_load(raw_value)


def main():
    parser = argparse.ArgumentParser(description="Train TRACE using its YAML config.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config value; may be repeated.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    apply_config_overrides(cfg, args.overrides)

    # Required sections
    paths = cfg.get("paths", {})
    find_marker_genes_paras = cfg.get("find_marker_genes_paras", {})
    pseudo_spot_simulation_paras = cfg.get("pseudo_spot_simulation_paras", {})
    data_normalization_paras = cfg.get("data_normalization_paras", {})
    feature_config = cfg.get("features", {}) or {}
    options = cfg.get("options", {})
    export_cfg = cfg.get("export", {})
    run_cfg = cfg.get("run", {}) or {}
    plot_cfg = cfg.get("plot", {}) or {}
    progress_cfg = cfg.get("progress", {}) or {}
    requested_pseudo_spots = int(pseudo_spot_simulation_paras.get('spot_num', 0))
    max_pseudo_spots = int(options.get('max_pseudo_spots', 100000))
    if (
        requested_pseudo_spots > max_pseudo_spots
        and not bool(options.get('allow_very_large_pseudo_spots', False))
    ):
        raise ValueError(
            f"pseudo_spot_simulation_paras.spot_num={requested_pseudo_spots:,} exceeds "
            f"the configured safety limit of {max_pseudo_spots:,}. This can exhaust RAM "
            "during pseudo-spot generation. Reduce spot_num or explicitly set "
            "options.allow_very_large_pseudo_spots=true if the allocation is intentional."
        )
    if not isinstance(feature_config, dict):
        raise TypeError("Config section 'features' must be a mapping")
    missing_inputs = [key for key in REQUIRED_PATH_KEYS if not paths.get(key)]
    if missing_inputs:
        raise KeyError(
            "Set all four TRACE input paths: "
            + ", ".join(f"paths.{key}" for key in missing_inputs)
        )
    for key in (*REQUIRED_PATH_KEYS, "spot_ground_truth_file"):
        if paths.get(key):
            input_path = Path(paths[key])
            if not input_path.is_absolute():
                input_path = config_path.parent / input_path
            paths[key] = str(input_path.resolve())

    seed, seed_mode = resolve_run_seed(run_cfg, cfg.get('split_seed'))
    split_seed = int(cfg.get("split_seed", seed) if cfg.get("split_seed") is not None else seed)
    seed_everything(seed, bool(run_cfg.get('deterministic', True)))
    device = resolve_device(options)
    torch.set_num_threads(max(1, int(options.get("torch_num_threads", min(4, os.cpu_count() or 1)))))
    print(f"Using device: {device}; reproducibility seed: {seed}")

    output_root = Path(paths.get("output_path", "output"))
    if not output_root.is_absolute():
        output_root = config_path.parent / output_root
    run_name = run_cfg.get('name')
    if run_name:
        run_name = str(run_name).format(seed=seed, seed_mode=seed_mode)
    outpath = output_root / str(run_name) if run_name else output_root
    outpath = str(outpath.resolve())
    os.makedirs(outpath, exist_ok=True)
    paths = dict(paths)
    paths['output_path'] = outpath
    cfg['paths'] = paths

    # ---------------- STEP 1: Preprocess ----------------
    results = preprocess_trace(
        input_paths=paths,
        output_path=outpath,
        marker_config=find_marker_genes_paras,
        pseudo_config=pseudo_spot_simulation_paras,
        normalization_config=data_normalization_paras,
        feature_config=feature_config,
        seed=seed,
    )

    Y_real_df = results.get("real_spots")
    Y_real_loc_df = results.get("real_coordinates")
    Y_pseudo_df = results.get("pseudo_spots")
    Y_all_df = results.get("all_spots")
    cell_types = results.get("cell_types")
    B_pseudo_gt_df = results.get("pseudo_dist")
    X_df = results.get("sc_ref")
    X = X_df.groupby("cell_type").mean().reindex(cell_types)
    if X.isna().any().any():
        raise ValueError("Single-cell reference means do not align with the selected cell types")

    # bring everything to torch tensors on the right device
    Y_real = torch.tensor(Y_real_df.values, dtype=torch.float32, device=device)
    Y_real_loc = torch.tensor(Y_real_loc_df.values, dtype=torch.float32, device=device)
    Y_pseudo = torch.tensor(Y_pseudo_df.values, dtype=torch.float32, device=device)
    Y_all = torch.tensor(Y_all_df.values, dtype=torch.float32, device=device)
    B_pseudo_gt = torch.tensor(B_pseudo_gt_df.values, dtype=torch.float32, device=device)
    X = torch.tensor(X.values, dtype=torch.float32, device=device)

    # ---------------- STEP 2: Build Graph ----------------
    num_real_spots = Y_real.shape[0]
    num_pseudo_spots = Y_pseudo.shape[0]
    num_total_spots = num_real_spots + num_pseudo_spots

    # Config knobs (add to your config as needed)
    gcfg = cfg.get("graph", {})
    defaults = {
        "alpha_spatial": 1.0,
        "alpha_expr_real": 1.0,
        "alpha_expr_pseudo": 1.0,
        "alpha_cross": 1.0,
        "k_real": 10,
        "k_pseudo": 10,
        "k_cross_real_to_pseudo": 10,
        "k_cross_pseudo_to_real": 10,
        "expr_thresh": 0.0,
        "mutual_knn": True,
        "use_spatial_knn": True,
        "adjacency_sigma": 2.0,
        "adjacency_threshold": 1e-4,
    }
    for k, v in defaults.items():
        gcfg.setdefault(k, v)
    gcfg.setdefault('n_jobs', options.get('n_jobs', -1))



    A_combined, edge_index_combined, edge_weight_combined = build_graphs_and_edges(
        Y_real=Y_real,
        Y_pseudo=Y_pseudo,
        Y_real_loc=Y_real_loc,
        device=device,
        gcfg=gcfg,
    )


    # ---------------- STEP 3: Define splits ----------------
    real_indices = np.arange(num_real_spots)
    pseudo_indices = np.arange(num_real_spots, num_total_spots)

    np.random.seed(split_seed)
    random.seed(split_seed)
    np.random.shuffle(real_indices)

    print(f"\nData Split Seed: {split_seed}\n")
    val_size = int(num_real_spots * cfg["val_size"])
    val_idx = real_indices[:val_size]

    train_mask = torch.ones(num_total_spots, dtype=torch.bool, device=device)           # train_real + pseudo
    train_mask[val_idx] = False

    val_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)            # only real
    val_mask[val_idx] = True

    pseudo_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)         # only pseudo
    pseudo_mask[pseudo_indices] = True

    train_real_mask = train_mask & ~pseudo_mask                                         # only train real 
    real_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
    real_mask[:num_real_spots] = True

    masks = {
        "train": train_mask,                # train set with both real + pseudo spots
        "train_real": train_real_mask,      # train set with only real spots
        "val": val_mask,                    # validation set with only real spots
        "real": real_mask,                  # all real spots (train_real + val)
        "pseudo": pseudo_mask,              # all pseudo spots  
    }

    print(f"Number of total training spots: {train_mask.sum().item()}")
    print(f"Number of real training spots: {train_real_mask.sum().item()}")
    print(f"Number of pseudo training spots: {pseudo_mask.sum().item()}")
    print(f"Number of validation spots: {val_mask.sum().item()}")


    # ---------------- STEP 4: Define model ----------------
    input_dim = Y_all.shape[1]
    n_celltypes = len(cell_types)
    model = TRACE(input_dim, cfg["hidden_dim"], cfg["latent_dim"],
                    n_celltypes, tau=cfg["tau"], drop=cfg["dropout_rate"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["l2_lambda"]))

    # ---------------- STEP 5: Training ----------------
    total_loss_history = []
    recon_loss_history = []
    kl_lat_loss_history = []
    kl_mix_loss_history = []
    smooth_loss_history = []
    entropy_loss_history = []
    align_loss_history = []
    l1_loss_history = []
    val_loss_history = []
    show_epoch_progress = bool(progress_cfg.get("enabled", True))
    epoch_iterator = trange(
        int(cfg["n_epochs"]),
        desc=f"Training seed {seed}",
        unit="epoch",
        dynamic_ncols=True,
        disable=not show_epoch_progress,
    )
    for epoch in epoch_iterator:
        model.train()
        optimizer.zero_grad()

        # Forward
        Y_hat_all, mu_all, logvar_all, B_pred_all, *rest = model(
            Y_all, edge_index_combined, edge_weight_combined, X
        )

        # Y_hat_all, mu_all, logvar_all, _, B_pred_all, *rest = model(
        #     Y_all, edge_index_combined, edge_weight_combined, X
        # )
        # B_pred_all = B_pred_all.exp()

        losses = vae_loss(
            Y_all=Y_all,
            Y_hat_all=Y_hat_all,
            mu=mu_all,
            logvar=logvar_all,
            A_all=A_combined,
            B_all=B_pred_all,
            B_pseudo_gt=B_pseudo_gt,
            masks=masks,
            lambda_recon=float(cfg["loss_recon"]),
            lambda_kl_lat=float(cfg["loss_kl_lat"]),
            lambda_kl_mix=float(cfg['loss_kl_mix']),
            lambda_smooth=float(cfg["loss_smooth"]),
            lambda_ent=float(cfg["loss_ent"]),
            lambda_align=float(cfg['loss_align']),
            lambda_l1=float(cfg['loss_l1']),
            evaluate=False
        )

        loss = losses["total"]
        loss.backward()
        optimizer.step()
        total_loss_history.append(loss.item())
        recon_loss_history.append(cfg["loss_recon"] * losses["recon"].item())
        kl_lat_loss_history.append(cfg["loss_kl_lat"] * losses["kl_lat"].item())
        kl_mix_loss_history.append(cfg['loss_kl_mix'] * losses.get("kl_mix", 0.0).item())
        smooth_loss_history.append(cfg["loss_smooth"] * losses["smooth"].item())
        entropy_loss_history.append(cfg["loss_ent"] * losses["entropy"].item())
        align_loss_history.append(cfg['loss_align'] * losses.get("align", 0.0).item())
        l1_loss_history.append(cfg['loss_l1'] * losses.get("l1", 0.0).item())

        # ---------------- Validation ----------------
        model.eval()
        with torch.no_grad():
            Y_hat_all_eval, mu_all_eval, logvar_all_eval, B_pred_all_eval, *rest = model(
                Y_all, edge_index_combined, edge_weight_combined, X
            )

            val_losses = vae_loss(
                Y_all=Y_all,
                Y_hat_all=Y_hat_all_eval,
                mu=mu_all_eval,
                logvar=logvar_all_eval,
                A_all=A_combined,
                B_all=B_pred_all_eval,
                B_pseudo_gt=B_pseudo_gt,
                masks=masks,
                lambda_recon=float(cfg["loss_recon"]),
                lambda_kl_lat=float(cfg["loss_kl_lat"]),
                lambda_kl_mix=float(cfg['loss_kl_mix']),
                lambda_smooth=float(cfg["loss_smooth"]),
                lambda_ent=float(cfg["loss_ent"]),
                lambda_align=float(cfg['loss_align']),
                lambda_l1=float(cfg['loss_l1']),
                evaluate=True
            )

            val_loss = val_losses["total"]
            val_loss_history.append(val_loss.item())

        if show_epoch_progress:
            epoch_iterator.set_postfix(
                train=f"{loss.item():.4f}",
                val=f"{val_loss.item():.4f}",
                refresh=True,
            )

        if not show_epoch_progress and ((epoch + 1) % 25 == 0 or epoch == 0):
            print(
                f"Epoch {epoch+1:03d}: "
                f"Train={loss.item():.4f} "
                f"(recon={losses['recon']:.4f}, "
                f"kl_lat={losses['kl_lat']:.4f}, "
                f"kl_mix={losses.get('kl_mix', 0.0):.4f}, "
                f"smooth={losses['smooth']:.4f}, "
                f"ent={losses['entropy']:.4f} "
                f'align={losses.get("align", 0.0):.4f}, '
                f'l1={losses.get("l1", 0.0):.4f}) '

                f"Val={val_loss.item():.4f} "
            )

    # save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history, label="Train (Real + Pseudo)", linewidth=4)
    plt.plot(val_loss_history, label="Val (Real)", linewidth=4)
    plt.plot(recon_loss_history, label="Reconstruction", linestyle='--')
    plt.plot(kl_lat_loss_history, label="KL Latent", linestyle='--')
    plt.plot(kl_mix_loss_history, label="KL Mixture", linestyle='--')
    plt.plot(smooth_loss_history, label="Graph Smoothness", linestyle='--')
    plt.plot(entropy_loss_history, label="Entropy", linestyle='--')
    plt.plot(align_loss_history, label="OT Alignment", linestyle='--')
    plt.plot(l1_loss_history, label="L1 Sparsity", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Semi-Supervised Training Loss")
    # plt.savefig(os.path.join(outpath, "loss_curves.svg"), format="svg", dpi=300)
    plt.savefig(os.path.join(outpath, "loss_curves.png"), dpi=300)
    plt.close()

    # ---------------- Evaluation ----------------
    print("\n### Final Evaluation ###")
    model.eval()
    with torch.no_grad():
        _, mu_all_final, _, B_pred_all, *rest = model(
            Y_all, edge_index_combined, edge_weight_combined, X
        )

    B_pred_all = B_pred_all[real_mask]
    mu_real_final = mu_all_final[real_mask].cpu().numpy()
    real_spot_names = Y_real_df.index.to_list()

    if export_cfg.get("save_real_latent_mu", False):
        latent_columns = [f"latent_mu_{i}" for i in range(mu_real_final.shape[1])]
        latent_mu_df = pd.DataFrame(mu_real_final, index=real_spot_names, columns=latent_columns)
        latent_mu_path = os.path.join(
            outpath,
            export_cfg.get("real_latent_mu_filename", "real_spot_latent_mu.pkl"),
        )
        latent_mu_df.to_pickle(latent_mu_path)
        print(f"Saved final real-spot latent mu to: {latent_mu_path}")

        if export_cfg.get("save_real_spot_names", False):
            real_spot_names_path = os.path.join(
                outpath,
                export_cfg.get("real_spot_names_filename", "real_spot_names.pkl"),
            )
            with open(real_spot_names_path, "wb") as f:
                pickle.dump(real_spot_names, f)
            print(f"Saved ordered real-spot names to: {real_spot_names_path}")

    y_pred = B_pred_all.cpu().numpy()
    predicted_proportions_path = os.path.join(
        outpath, export_cfg.get('predicted_proportions_filename', 'predicted_proportions.csv')
    )
    B_df = pd.DataFrame(y_pred, index=real_spot_names, columns=cell_types)
    B_df.to_csv(predicted_proportions_path)
    print(f"Saved predicted proportions to: {predicted_proportions_path}")
    model_path = os.path.join(outpath, "TRACE_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cell_types": [str(cell_type) for cell_type in cell_types],
            "genes": list(Y_all_df.columns),
            "config": cfg,
        },
        model_path,
    )
    print(f"Saved trained TRACE model to: {model_path}")

    resolved_config_path = os.path.join(outpath, "resolved_config.yaml")
    cfg_to_save = dict(cfg)
    cfg_to_save['paths'] = dict(cfg.get('paths', {}) or {})
    cfg_to_save['paths']['output_path'] = str(output_root.resolve())
    cfg_to_save["split_seed"] = split_seed
    cfg_to_save['run'] = dict(cfg.get('run', {}) or {})
    cfg_to_save['run']['requested_seed_mode'] = seed_mode
    # The resolved config is intentionally fixed so it can reproduce a run
    # that originally requested a fresh random seed.
    cfg_to_save['run']['seed_mode'] = 'fixed'
    cfg_to_save['run']['seed'] = seed
    cfg_to_save['run']['name'] = run_name
    with open(resolved_config_path, "w") as f:
        yaml.safe_dump(cfg_to_save, f, sort_keys=False)
    print(f"Saved resolved config to: {resolved_config_path}")

    G_all_df = results.get("ground_truth")
    if G_all_df is not None:
        G_all_df = G_all_df.loc[real_spot_names]
        # Convert to numpy for evaluation and saving
        y_true = G_all_df.loc[:, list(cell_types)].to_numpy()

        # Evaluate metrics
        results = evaluate_all(y_true, y_pred)
        print("\nEvaluation Results:")
        for k, v in results.items():
            print(f"{k:12s}: {v:.4f}")

        # append config parameters to results
        for k, v in cfg.items():
            results[k] = v
        results["split_seed"] = split_seed

        # append if results already exist
        metrics_file = os.path.join(outpath, "metrics.csv")
        if os.path.exists(metrics_file):
            existing_results = pd.read_csv(metrics_file)
            results_df = pd.DataFrame([results])
            results_df = pd.concat([existing_results, results_df], ignore_index=True)
            results_df.to_csv(metrics_file, index=False)
        else:
            pd.DataFrame([results]).to_csv(os.path.join(outpath, "metrics.csv"), index=False)

    else:
        print("Skipping evaluation metrics because the dataset has no spot ground truth")

    if bool(plot_cfg.get('enabled', True)):
        colors_file = plot_cfg.get('celltype_colors_file')
        color_map = resolve_celltype_colors(cell_types, colors_file=colors_file)
        pd.DataFrame(
            [(label, color_map[label]) for label in cell_types],
            columns=['cell_type', 'color'],
        ).to_csv(os.path.join(outpath, 'celltype_colors.tsv'), sep='\t', index=False)
        plot_path = os.path.join(
            outpath, plot_cfg.get('filename', 'predicted_spatial_pies.png')
        )
        plotted_spots, pie_radius = draw_spatial_pies(
            B_df,
            Y_real_loc_df,
            color_map,
            plot_path,
            plot_cfg,
            seed=seed,
        )
        print(
            f"Saved spatial pie chart to: {plot_path} "
            f"({plotted_spots:,} spots; radius={pie_radius:.4g})"
        )


if __name__ == "__main__":
    main()
