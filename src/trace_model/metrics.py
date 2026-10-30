import numpy as np
from numpy.linalg import norm
from scipy.special import rel_entr
from scipy.stats import kendalltau, pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def ccc(y_true, y_pred, eps=1e-8):
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    mean_true, mean_pred = np.mean(yt), np.mean(yp)
    var_true, var_pred = np.var(yt), np.var(yp)
    cov = np.mean((yt - mean_true) * (yp - mean_pred))
    return float((2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + eps))


def jsd(p, q, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None) + eps
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, None) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m)))


def hellinger(p, q, eps=0.0):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None) + eps
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, None) + eps
    p = p / (p.sum() if p.sum() > 0 else 1.0)
    q = q / (q.sum() if q.sum() > 0 else 1.0)
    return float((1.0 / np.sqrt(2.0)) * norm(np.sqrt(p) - np.sqrt(q)))


def cosine_sim(p, q, eps=1e-10):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.dot(p, q) / (norm(p) * norm(q) + eps))


def emd_on_celltype_index(p, q):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, None)
    p = p / (p.sum() if p.sum() > 0 else 1.0)
    q = q / (q.sum() if q.sum() > 0 else 1.0)
    x = np.arange(p.shape[0], dtype=np.float64)
    return float(wasserstein_distance(x, x, u_weights=p, v_weights=q))


def shannon_entropy(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None) + eps
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def support_metrics(p, q, threshold):
    sp = np.asarray(p, dtype=np.float64) > threshold
    sq = np.asarray(q, dtype=np.float64) > threshold
    tp = int(np.sum(sp & sq))
    fp = int(np.sum(~sp & sq))
    fn = int(np.sum(sp & ~sq))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return float(f1), float(jacc), float(prec), float(rec)


def dominance_profile_l1(p, q):
    ps = np.sort(np.asarray(p, dtype=np.float64))[::-1]
    qs = np.sort(np.asarray(q, dtype=np.float64))[::-1]
    cp = np.cumsum(ps)
    cq = np.cumsum(qs)
    return float(np.mean(np.abs(cp - cq)))


def rbo_rank_overlap(order_a, order_b, p=0.9):
    order_a = np.asarray(order_a)
    order_b = np.asarray(order_b)
    k = min(len(order_a), len(order_b))
    if k == 0:
        return 0.0

    seen_a = set()
    seen_b = set()
    summation = 0.0
    overlap = 0
    for depth in range(1, k + 1):
        seen_a.add(int(order_a[depth - 1]))
        seen_b.add(int(order_b[depth - 1]))
        overlap = len(seen_a & seen_b)
        summation += (overlap / depth) * (p ** (depth - 1))
    return float((1 - p) * summation + (overlap / k) * (p ** k))


def evaluate_all(P, Q, threshold=0.05, jsd_eps=1e-12, rbo_p=0.9):
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have the same shape, got {P.shape} vs {Q.shape}")
    if P.ndim != 2:
        raise ValueError(f"P and Q must be 2D arrays of shape (spots, celltypes), got ndim={P.ndim}")

    n, k = P.shape
    results = {}

    results["MSE"] = mse(P, Q)
    results["RMSE"] = float(np.sqrt(results["MSE"]))
    results["MAE"] = mae(P, Q)
    results["CCC"] = ccc(P, Q)

    jsds = np.array([jsd(P[i], Q[i], eps=jsd_eps) for i in range(n)], dtype=np.float64)
    hells = np.array([hellinger(P[i], Q[i]) for i in range(n)], dtype=np.float64)
    emds = np.array([emd_on_celltype_index(P[i], Q[i]) for i in range(n)], dtype=np.float64)

    spears = []
    pears = []
    ktaus = []
    cosines = []
    ent_abs = []
    dom_l1 = []

    for i in range(n):
        pi, qi = P[i], Q[i]
        spears.append(spearmanr(pi, qi)[0])
        pears.append(pearsonr(pi, qi)[0])
        ktaus.append(kendalltau(pi, qi)[0])
        cosines.append(cosine_sim(pi, qi))
        ent_abs.append(abs(shannon_entropy(pi, eps=jsd_eps) - shannon_entropy(qi, eps=jsd_eps)))
        dom_l1.append(dominance_profile_l1(pi, qi))

    results["JSD_spot_mean"] = float(np.mean(jsds))
    results["Hellinger_spot_mean"] = float(np.mean(hells))
    results["EMD_index_spot_mean"] = float(np.mean(emds))
    results["Cosine_spot_mean"] = float(np.mean(cosines))
    results["Pearson_spot_mean"] = float(np.nanmean(pears))
    results["Spearman_spot_mean"] = float(np.nanmean(spears))
    results["KendallTau_spot_mean"] = float(np.nanmean(ktaus))
    results["EntropyAbs_spot_mean"] = float(np.mean(ent_abs))
    results["DominanceProfileL1_spot_mean"] = float(np.mean(dom_l1))

    p_bar = P.mean(axis=0)
    q_bar = Q.mean(axis=0)
    results["JSD_global_meancomp"] = jsd(p_bar, q_bar, eps=jsd_eps)
    results["Hellinger_global_meancomp"] = hellinger(p_bar, q_bar)

    col_pears = []
    col_spears = []
    col_maes = []
    for j in range(k):
        pj, qj = P[:, j], Q[:, j]
        col_pears.append(pearsonr(pj, qj)[0])
        col_spears.append(spearmanr(pj, qj)[0])
        col_maes.append(float(mean_absolute_error(pj, qj)))

    results["Pearson_celltype_mean"] = float(np.nanmean(col_pears))
    results["Spearman_celltype_mean"] = float(np.nanmean(col_spears))
    results["MAE_celltype_mean"] = float(np.mean(col_maes))

    results["Top1_Accuracy"] = float(np.mean(np.argmax(P, axis=1) == np.argmax(Q, axis=1)))

    f1s = []
    jaccs = []
    precs = []
    recs = []
    rbos = []
    for i in range(n):
        f1, jacc, prec, rec = support_metrics(P[i], Q[i], threshold=threshold)
        f1s.append(f1)
        jaccs.append(jacc)
        precs.append(prec)
        recs.append(rec)
        order_p = np.argsort(-P[i], kind="mergesort")
        order_q = np.argsort(-Q[i], kind="mergesort")
        rbos.append(rbo_rank_overlap(order_p, order_q, p=rbo_p))

    results["SupportF1"] = float(np.mean(f1s))
    results["SupportJaccard"] = float(np.mean(jaccs))
    results["SupportPrecision"] = float(np.mean(precs))
    results["SupportRecall"] = float(np.mean(recs))
    results["RBO"] = float(np.mean(rbos))

    # Legacy aliases kept for compatibility with existing v4 scripts.
    results["JSD"] = results["JSD_spot_mean"]
    results["Hellinger"] = results["Hellinger_spot_mean"]
    results["EMD"] = results["EMD_index_spot_mean"]
    results["Spearman"] = results["Spearman_spot_mean"]
    results["Pearson"] = results["Pearson_spot_mean"]
    results["Cosine"] = results["Cosine_spot_mean"]

    return results
