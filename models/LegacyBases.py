"""User's two earlier wind-power projects, ported for the X2 base screening (Claude, 2026-11-16).

Sources (read-only, D:/Users/XT/Desktop/nuist/lunwen/):
  A  风力发电预测：一种结合混合模型和对抗关系训练的时间域泛化方法 / models/da_tcn_gru_v2.py (+ train_v2.py config)
  B  基于CNN-LSTM-LightGBM的短期风电功率预测 / src/train_zyx_ws.py (Keras CNN / BiLSTM + LightGBM stacking)

Interface (same as G1TSMixerModel, so scripts/train_g1.fit_one / predict can be reused):
  x [B, 24, C] in the frozen fit-period normalisation (power last) -> [B, 4] normalised power.
  `arm = "N"` makes train_g1.predict feed x50; fit_station_normalization is a no-op.

Changes from the originals (registered in STAGE_PLAN X2 and DECISIONS):
  * 72/6 -> 24/4 steps; no [0, 1] output clamp (targets are standardised, not min-max).
  * Same inputs as every other arm (x50 incl. upstream + GFS; HoT adds the retrofit channel); no calendar features.
  * DA: the GRL coefficient is applied ONCE and the domain loss enters with weight 1 (standard DANN).
    The original multiplied lambda in the GRL and again in the total loss, so features saw -lambda^2
    (0.005 -> 2.5e-5) and the discriminator learnt at weight lambda.  This port is "DA (standard GRL)",
    not a bit-exact reproduction.
  * Keras recurrent_dropout / l1_l2 kernel regularisers are dropped (cuDNN LSTM; AdamW weight decay is shared).
  * Stacking (fusion) is causal: the meta-learner sees only calibration-segment predictions of members fitted
    on the fit period; forward-chaining over contiguous issue-time blocks replaces in-sample / KFold stacking.
"""
from __future__ import annotations

import numpy as np

try:  # numpy-only helpers below must import without torch (sandbox, CPU stacking)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None

ARMS = ("TCNGRU", "DATCNGRU", "CNN", "BILSTM")
TCN_CONFIG = dict(channels=(64, 128, 128, 64), kernel_size=7, dropout=0.1)  # train_v2.py CONFIG


# ----------------------------------------------------------------------------- numpy helpers
def lambda_schedule(epoch, max_epochs, lam_end, warmup=0.4):
    """Original schedule (0 for the first 40% epochs, then linear), epoch 1-based as in fit_one.
    Reaches lam_end at the last epoch.  Applied once, inside the GRL."""
    warm = int(warmup * max_epochs)
    if epoch <= warm:
        return 0.0
    return float(lam_end) * (epoch - warm) / (max_epochs - warm)


def time_domains(issue, k=3):
    """Original domain labels: the fit period cut into k equal-length contiguous time segments (by issue time)."""
    t = np.asarray(issue).astype("datetime64[s]").astype(np.int64)
    lo, hi = t.min(), t.max()
    if hi <= lo:
        raise ValueError("Domain split needs a non-empty time span")
    edges = lo + (hi - lo) * np.arange(1, k) / k
    return np.searchsorted(edges, t, side="right").astype(np.int64)


def forward_blocks(issue, n_blocks=3):
    """Contiguous issue-time blocks 0..n_blocks-1 (all windows of one issue time share a block)."""
    t = np.asarray(issue).astype("datetime64[s]").astype(np.int64)
    uniq = np.unique(t)
    if len(uniq) < n_blocks:
        raise ValueError("Fewer issue times than blocks")
    edges = uniq[np.linspace(0, len(uniq), n_blocks + 1).astype(int)[1:-1]]
    return np.searchsorted(edges, t, side="right")


def _ridge(X, y, alpha):
    xm, ym = X.mean(0), y.mean()
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, float(ym - xm @ w)


def fit_stacking(members, truth, issue, scale, alphas=(0.1, 1, 10, 20, 50), n_blocks=3):
    """Ridge meta-learner per horizon, fitted on the calibration segment only.

    members [n, m, 4] kW (m base models, already fitted on the fit period), truth [n, 4] kW, issue [n].
    Alpha (original grid) is chosen by forward-chaining MSE: fold b trains on blocks < b, scores block b.
    Values are divided by `scale` (rated kW) so alpha is unit-free.  Ties keep the earlier (smaller) alpha.
    """
    P = np.asarray(members, float) / scale
    Y = np.asarray(truth, float) / scale
    if P.ndim != 3 or Y.shape != (P.shape[0], P.shape[2]) or len(issue) != len(P):
        raise ValueError("members [n, m, H], truth [n, H] and issue [n] required")
    if not (np.isfinite(P).all() and np.isfinite(Y).all()):
        raise ValueError("Finite calibration predictions and truth required")
    block = forward_blocks(issue, n_blocks)
    horizons = []
    for h in range(Y.shape[1]):
        X, y = P[:, :, h], Y[:, h]
        scores = []
        for a in alphas:
            err = []
            for b in range(1, n_blocks):
                tr, va = block < b, block == b
                w, c = _ridge(X[tr], y[tr], a)
                err.append(float(np.mean((X[va] @ w + c - y[va]) ** 2)))
            scores.append(float(np.mean(err)))
        a = alphas[int(np.argmin(scores))]
        w, c = _ridge(X, y, a)
        horizons.append(dict(alpha=float(a), weights=w.tolist(), intercept=c, cv_mse=scores))
    return dict(scale=float(scale), members=int(P.shape[1]), n_blocks=int(n_blocks), horizons=horizons)


def apply_stacking(members, model):
    P = np.asarray(members, float) / model["scale"]
    if P.ndim != 3 or P.shape[1] != model["members"] or P.shape[2] != len(model["horizons"]):
        raise ValueError("Member layout differs from the fitted meta-learner")
    cols = [P[:, :, h] @ np.asarray(m["weights"]) + m["intercept"] for h, m in enumerate(model["horizons"])]
    return np.stack(cols, axis=1) * model["scale"]


# ----------------------------------------------------------------------------- torch models
if torch is not None:

    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = float(lam)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.lam * grad, None

    def grad_reverse(x, lam):
        return _GradReverse.apply(x, lam)

    def _weight_norm(conv):
        try:
            return nn.utils.parametrizations.weight_norm(conv)
        except AttributeError:  # older torch
            return nn.utils.weight_norm(conv)

    class _Chomp(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return x[:, :, :-self.n].contiguous() if self.n else x

    class _TemporalBlock(nn.Module):
        """da_tcn_gru_v2.TemporalBlock; N(0, 0.01) init is applied before weight_norm so it persists."""

        def __init__(self, cin, cout, k, dilation, dropout):
            super().__init__()
            pad = (k - 1) * dilation
            c1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
            c2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation)
            c1.weight.data.normal_(0, 0.01)
            c2.weight.data.normal_(0, 0.01)
            self.net = nn.Sequential(_weight_norm(c1), _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                                     _weight_norm(c2), _Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
            self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None
            if self.down is not None:
                self.down.weight.data.normal_(0, 0.01)

        def forward(self, x):
            res = x if self.down is None else self.down(x)
            return F.relu(self.net(x) + res)

    class _LegacyBase(nn.Module):
        arm = "N"  # train_g1.predict feeds x50 unless arm == "B"

        def fit_station_normalization(self, batches):  # interface no-op: no station block
            return None

    class TCNGRU(_LegacyBase):
        """Project A feature extractor (TCN -> GRU, last step) + power head (MLP h -> h/2 -> 4)."""

        def __init__(self, c_in, channels=TCN_CONFIG["channels"], kernel_size=TCN_CONFIG["kernel_size"],
                     dropout=TCN_CONFIG["dropout"], pred_len=4):
            super().__init__()
            self.tcn = nn.Sequential(*[_TemporalBlock(c_in if i == 0 else channels[i - 1], c, kernel_size, 2 ** i, dropout)
                                       for i, c in enumerate(channels)])
            h = channels[-1]
            self.gru = nn.GRU(h, h, batch_first=True)
            self.head = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h // 2, pred_len))
            self.hidden = h

        def features(self, x):
            z = self.tcn(x.transpose(1, 2))
            out, _ = self.gru(z.transpose(1, 2))
            return out[:, -1]

        def forward(self, x, batch=None):
            return self.head(self.features(x))

    class DATCNGRU(TCNGRU):
        """Project A with the domain branch; forward() is the power path only (evaluation identical to TCNGRU)."""

        def __init__(self, c_in, num_domains=3, **kw):
            super().__init__(c_in, **kw)
            self.discriminator = nn.Sequential(nn.Linear(self.hidden, 32), nn.ReLU(), nn.Dropout(0.5),
                                               nn.Linear(32, num_domains))

        def forward_with_domain(self, x, lam):
            f = self.features(x)
            return self.head(f), self.discriminator(grad_reverse(f, lam))

    def da_loss(model, x, target, domain, lam, mae_weight=0.8, mse_weight=0.2):
        """Total = task (same 0.8 MAE + 0.2 MSE as every arm) + CE(domain); lam only inside the GRL."""
        pred, logits = model.forward_with_domain(x, lam)
        err = pred.reshape(target.shape) - target
        task = mae_weight * err.abs().mean() + mse_weight * err.square().mean()
        dom = F.cross_entropy(logits, domain)
        return task + dom, task.detach(), dom.detach()

    class CNNAttn(_LegacyBase):
        """Project B build_cnn_model: 3 x (Conv-ReLU-BN-MaxPool-Dropout), sigmoid conv attention, GAP, MLP."""

        def __init__(self, c_in, pred_len=4):
            super().__init__()

            def block(i, o, k):
                return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.ReLU(), nn.BatchNorm1d(o),
                                     nn.MaxPool1d(2), nn.Dropout(0.25))

            self.blocks = nn.Sequential(block(c_in, 64, 5), block(64, 128, 3), block(128, 256, 3))
            self.attn = nn.Conv1d(256, 256, 3, padding=1)
            self.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, pred_len))

        def forward(self, x, batch=None):
            z = self.blocks(x.transpose(1, 2))
            z = z * torch.sigmoid(self.attn(z))
            return self.fc(z.mean(-1))

    class BiLSTMAttn(_LegacyBase):
        """Project B build_Bilstm_model: BiLSTM(128)-LN-Drop, BiLSTM(192)-LN-Drop, tanh/softmax time attention,
        GAP of the weighted sequence (as in Keras), MLP.  Bidirectional only within the past input window."""

        def __init__(self, c_in, pred_len=4):
            super().__init__()
            self.l1 = nn.LSTM(c_in, 128, batch_first=True, bidirectional=True)
            self.n1, self.d1 = nn.LayerNorm(256), nn.Dropout(0.25)
            self.l2 = nn.LSTM(256, 192, batch_first=True, bidirectional=True)
            self.n2, self.d2 = nn.LayerNorm(384), nn.Dropout(0.25)
            self.score = nn.Linear(384, 1)
            self.fc = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, pred_len))

        def forward(self, x, batch=None):
            z, _ = self.l1(x)
            z = self.d1(self.n1(z))
            z, _ = self.l2(z)
            z = self.d2(self.n2(z))
            w = torch.softmax(torch.tanh(self.score(z)).squeeze(-1), dim=1).unsqueeze(-1)
            return self.fc((z * w).mean(1))

    def build_legacy(arm, c_in, pred_len=4, num_domains=3):
        if arm == "TCNGRU":
            return TCNGRU(c_in, pred_len=pred_len)
        if arm == "DATCNGRU":
            return DATCNGRU(c_in, num_domains=num_domains, pred_len=pred_len)
        if arm == "CNN":
            return CNNAttn(c_in, pred_len=pred_len)
        if arm == "BILSTM":
            return BiLSTMAttn(c_in, pred_len=pred_len)
        raise ValueError(f"arm must be one of {ARMS}")
