"""Thay thế scikit-learn bằng numpy thuần khi máy không import được sklearn (ví dụ Device Guard / WDAC chặn DLL).

Dùng:  from .skfallback import roc_auc_score, average_precision_score, roc_curve, LogisticRegression, make_pipeline,
                              StandardScaler, cross_val_predict
Nếu sklearn import được thì các tên này chính là của sklearn; nếu không thì là bản numpy dưới đây (cùng kết quả cho
AUROC/AP/ROC; logistic regression giải bằng gradient descent + L2, đủ cho probe tuyến tính trên đặc trưng đã chuẩn hoá).
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: F401
    from sklearn.model_selection import cross_val_predict  # noqa: F401
    from sklearn.pipeline import make_pipeline  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    HAVE_SKLEARN = True
except Exception:  # ImportError hoặc OSError (DLL bị chặn)
    HAVE_SKLEARN = False

    def roc_curve(y, p):
        y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
        order = np.argsort(-p, kind="mergesort")
        y, p = y[order], p[order]
        distinct = np.where(np.diff(p))[0]
        idx = np.r_[distinct, y.size - 1]
        tps = np.cumsum(y)[idx]; fps = (1 + idx) - tps
        tps = np.r_[0, tps]; fps = np.r_[0, fps]; thr = np.r_[np.inf, p[idx]]
        P, N = max(tps[-1], 1), max(fps[-1], 1)
        return fps / N, tps / P, thr

    def roc_auc_score(y, p):
        y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
        n1, n0 = int((y == 1).sum()), int((y == 0).sum())
        if n1 == 0 or n0 == 0:
            raise ValueError("Cần cả hai lớp")
        # AUROC = (tổng hạng của lớp dương - n1(n1+1)/2) / (n1 n0), hạng trung bình cho giá trị trùng
        order = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
        ps = p[order]; i = 0
        while i < len(ps):
            j = i
            while j + 1 < len(ps) and ps[j + 1] == ps[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def average_precision_score(y, p):
        """Như sklearn: các giá trị p trùng nhau gộp thành một ngưỡng."""
        y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
        order = np.argsort(-p, kind="mergesort"); y, p = y[order], p[order]
        idx = np.r_[np.where(np.diff(p))[0], len(p) - 1]           # vị trí cuối của mỗi nhóm ngưỡng
        tp = np.cumsum(y)[idx]; prec = tp / (idx + 1); rec = tp / max(tp[-1], 1)
        rec_prev = np.r_[0, rec[:-1]]
        return float(((rec - rec_prev) * prec).sum())

    class StandardScaler:
        def fit(self, X, y=None):
            X = np.asarray(X, dtype=np.float64); self.mean_ = X.mean(0); self.scale_ = X.std(0) + 1e-8; return self
        def transform(self, X): return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        def fit_transform(self, X, y=None): return self.fit(X).transform(X)

    class LogisticRegression:
        """Hồi quy logistic L2 (C = 1/lambda), gradient descent có momentum trên dữ liệu đã chuẩn hoá."""
        def __init__(self, max_iter=1000, C=1.0, lr=0.1, **kw):
            self.max_iter, self.C, self.lr = max_iter, C, lr
        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
            n, d = X.shape; w = np.zeros(d); b = 0.0; vw = np.zeros(d); vb = 0.0; lam = 1.0 / (self.C * n)
            for _ in range(self.max_iter):
                z = X @ w + b; s = 1 / (1 + np.exp(-z)); g = s - y
                gw = X.T @ g / n + lam * w; gb = g.mean()
                vw = 0.9 * vw + gw; vb = 0.9 * vb + gb
                w -= self.lr * vw; b -= self.lr * vb
                if np.abs(gw).max() < 1e-6:
                    break
            self.coef_, self.intercept_ = w[None, :], np.array([b]); return self
        def decision_function(self, X): return np.asarray(X, dtype=np.float64) @ self.coef_[0] + self.intercept_[0]
        def predict_proba(self, X):
            p = 1 / (1 + np.exp(-self.decision_function(X))); return np.stack([1 - p, p], 1)
        def predict(self, X): return (self.decision_function(X) > 0).astype(int)

    class _Pipeline:
        def __init__(self, steps): self.steps = list(steps)
        def fit(self, X, y):
            for s in self.steps[:-1]:
                X = s.fit_transform(X, y)
            self.steps[-1].fit(X, y); return self
        def _tx(self, X):
            for s in self.steps[:-1]:
                X = s.transform(X)
            return X
        def predict_proba(self, X): return self.steps[-1].predict_proba(self._tx(X))
        def decision_function(self, X): return self.steps[-1].decision_function(self._tx(X))
        def predict(self, X): return self.steps[-1].predict(self._tx(X))

    def make_pipeline(*steps): return _Pipeline(steps)


# ---- scipy.stats thay thế (rankdata, spearmanr, norm.cdf) — scipy cũng có thể bị chặn DLL
try:
    from scipy.stats import rankdata, spearmanr  # noqa: F401
    from scipy.stats import norm as _norm
    norm_cdf = _norm.cdf
except Exception:
    import math

    def rankdata(a):
        a = np.asarray(a, dtype=float); order = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); s_ = a[order]
        i = 0
        while i < len(s_):
            j = i
            while j + 1 < len(s_) and s_[j + 1] == s_[i]:
                j += 1
            r[order[i:j + 1]] = (i + j + 2) / 2.0; i = j + 1
        return r

    def spearmanr(a, b):
        ra, rb = rankdata(a), rankdata(b)
        ra = ra - ra.mean(); rb = rb - rb.mean()
        rho = float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12))
        n = len(ra)
        # p-value xấp xỉ qua t-phân phối -> dùng chuẩn khi n > 10
        t = rho * np.sqrt(max(n - 2, 1) / max(1 - rho ** 2, 1e-12))
        p = 2 * (1 - norm_cdf(abs(t)))
        class R:  # giống scipy: .statistic / .correlation / .pvalue và unpack (rho, p)
            statistic = correlation = rho; pvalue = p
            def __iter__(self): return iter((rho, p))
        return R()

    def norm_cdf(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))

    def cross_val_predict(est, X, y, cv=5, method="predict_proba"):
        X, y = np.asarray(X), np.asarray(y); n = len(y); out = None
        folds = np.array_split(np.random.default_rng(0).permutation(n), cv)
        import copy
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            m = copy.deepcopy(est).fit(X[tr], y[tr]); pred = getattr(m, method)(X[te])
            if out is None:
                out = np.zeros((n,) + np.asarray(pred).shape[1:])
            out[te] = pred
        return out
