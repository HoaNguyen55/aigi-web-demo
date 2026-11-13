"""Mô hình (mục 8): detector có cổng cho từng nhóm đặc trưng + các thân mạnh từ khảo sát.

train.trunk:
- "cnn"            : CNN nhỏ kiểu NPR trên đặc trưng đã qua cổng (mô hình gốc cần cải thiện)
- "vit"            : ViT (scratch/pretrained) trên đặc trưng đã qua cổng
- "timm:<tên>"     : thân timm bất kỳ trên đặc trưng đã qua cổng, ví dụ timm:convnext_tiny.fb_in22k, timm:resnet50
- "rgb:<tên>"      : mô hình timm pretrained trên ảnh RGB (không cổng), ví dụ
                       rgb:resnet50.a1_in1k                       (~ CNNSpot)
                       rgb:convnext_tiny.fb_in22k                 (~ nhánh ConvNeXt của AIDE)
                       rgb:vit_small_patch14_dinov2.lvd142m       (DINOv2, "Brought a Gun..." 2025)
                       rgb:vit_small_patch16_224.augreg_in21k     (= "vit_rgb")
- "clip_ft"        : CLIP fine-tune N khối cuối (train.clip_ft_blocks; 0 = linear probe ~ UnivFD),
                     xấp xỉ hướng CLIP của C2P-CLIP / Effort / FatFormer (không phải tái hiện đầy đủ)
train.freeze_backbone: true -> chỉ train lớp cuối (linear probe) cho "rgb:<tên>".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .features import Group, compute_groups, fft_radial


class Block(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        self.skip = nn.Identity() if (c_in == c_out and stride == 1) else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        return torch.relu(self.f(x) + self.skip(x))


class CNNTrunk(nn.Module):
    out_dim = 128

    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(Block(width, 64, 2), Block(64, 128, 2), Block(128, 128, 2),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, h):
        return self.net(h)


def _create_vit(model_name, pretrained, **kw):
    import timm
    try:
        return timm.create_model(model_name, pretrained=pretrained, **kw)
    except Exception as e:  # một số bản timm không nạp được trọng số khi đổi patch/in_chans -> nạp tay phần thân
        if not pretrained:
            raise
        print(f"[ViT] timm không nạp trực tiếp ({e}); nạp thân ViT, học lại patch embed + pos embed")
        m = timm.create_model(model_name, pretrained=False, **kw)
        ref = timm.create_model(model_name, pretrained=True, num_classes=kw.get("num_classes", 0))
        sd = {k: v for k, v in ref.state_dict().items()
              if not k.startswith(("patch_embed", "pos_embed")) and k in m.state_dict()
              and m.state_dict()[k].shape == v.shape}
        m.load_state_dict(sd, strict=False)
        return m


class ViTTrunk(nn.Module):
    def __init__(self, width=32, img=64, patch=4, pretrained=False, model_name="vit_small_patch16_224.augreg_in21k"):
        super().__init__()
        self.vit = _create_vit(model_name, pretrained, img_size=img, patch_size=patch, in_chans=width, num_classes=0)
        self.out_dim = self.vit.num_features

    def forward(self, h):
        return self.vit(h)


class TimmTrunk(nn.Module):
    """Thân timm bất kỳ nhận bản đồ đặc trưng [B,width,H/4,W/4]."""

    def __init__(self, name, width=32, img=64, pretrained=False):
        super().__init__()
        import timm
        kw = dict(pretrained=pretrained, in_chans=width, num_classes=0)
        try:
            self.m = timm.create_model(name, img_size=img, **kw)
        except TypeError:
            self.m = timm.create_model(name, **kw)
        self.out_dim = self.m.num_features

    def forward(self, h):
        return self.m(h)


class GatedDetector(nn.Module):
    """Cổng sigmoid(α_g) nhân vào từng nhóm. use_gates=False: cổng cố định = 1 (giai đoạn train gọn)."""

    def __init__(self, spec: list[Group], trunk="cnn", width=32, pretrained=False, use_gates=True, crop=256):
        super().__init__()
        self.spec = spec
        self.sp_groups = [g for g in spec if g.kind == "sp"]
        self.vec_groups = [g for g in spec if g.kind == "vec"]
        self.use_gates = use_gates
        self.stems = nn.ModuleList([nn.Conv2d(g.dim, width, 1) for g in self.sp_groups])
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(g.dim, 64), nn.ReLU(True)) for g in self.vec_groups])
        self.gate = nn.Parameter(torch.full((len(spec),), 2.0))  # sigmoid(2) ~ 0.88
        if self.sp_groups:
            if trunk == "cnn":
                self.trunk = CNNTrunk(width)
            elif trunk == "vit":
                self.trunk = ViTTrunk(width, img=crop // 4, patch=4, pretrained=pretrained)
            elif trunk.startswith("timm:"):
                self.trunk = TimmTrunk(trunk[5:], width, img=crop // 4, pretrained=pretrained)
            else:
                raise ValueError(f"trunk không hỗ trợ cho mô hình có cổng: {trunk}")
            feat_dim = self.trunk.out_dim
        else:
            self.trunk, feat_dim = None, 0
        self.head = nn.Linear(feat_dim + (64 if self.vec_groups else 0), 1)

    @property
    def needs_clip(self):
        return any(g.name == "clip" for g in self.spec)

    def gates(self):
        return torch.sigmoid(self.gate) if self.use_gates else torch.ones_like(self.gate)

    def gate_dict(self):
        """Mức mở cổng từng nhóm (chỉ có ý nghĩa ở giai đoạn select)."""
        if not self.use_gates:
            return {}
        return {g.name: float(v) for g, v in zip(self.spec, torch.sigmoid(self.gate).detach().cpu())}

    def embed(self, x, clip_emb=None):
        """Vector đặc trưng của 1 mảnh (trước lớp phân loại): [B, 384 + 64]."""
        sp, vec = compute_groups(x, self.spec, clip_emb)
        g = self.gates()
        feats = []
        if self.sp_groups:
            h = sum(g[i] * self.stems[i](sp[i]) for i in range(len(sp)))
            feats.append(self.trunk(h))
        if self.vec_groups:
            o = len(self.sp_groups)
            feats.append(sum(g[o + j] * self.projs[j](vec[j]) for j in range(len(vec))))
        return torch.cat(feats, 1)

    def forward(self, x, clip_emb=None):
        return self.head(self.embed(x, clip_emb)).squeeze(1)


def segment_softmax(s: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """softmax của s trong từng nhóm seg (mỗi nhóm = các mảnh của 1 ảnh)."""
    m = torch.full((n,), float("-inf"), device=s.device, dtype=s.dtype).scatter_reduce(0, seg, s, reduce="amax")
    e = torch.exp(s - m[seg])
    return e / torch.zeros(n, device=s.device, dtype=s.dtype).index_add(0, seg, e)[seg]


class TileMIL(nn.Module):
    """Liên kết các mảnh 256x256 của CÙNG một ảnh (Multiple Instance Learning, gộp có chú ý, Ilse et al. ICML 2018).

    Ảnh kích thước bất kỳ -> K mảnh 256x256 (không resize). Mỗi mảnh -> vector h_i (GatedDetector.embed).
    Trọng số chú ý a_i = softmax_i( w^T [tanh(V h_i) * sigmoid(U h_i)] ), tính TRONG các mảnh của cùng ảnh.
    z = sum_i a_i h_i -> lớp phân loại -> 1 logit cho CẢ ẢNH. Loss tính trên nhãn của ảnh, nên mô hình tự học
    mảnh nào chứa bằng chứng (mảnh giàu dấu vết -> a_i lớn, mảnh phẳng -> a_i nhỏ).
    counts = số mảnh của từng ảnh trong batch (None: mỗi mẫu là 1 ảnh 1 mảnh)."""
    is_mil = True

    def __init__(self, base: "GatedDetector", attn_dim: int = 128):
        super().__init__()
        self.base = base
        d = base.head.in_features
        self.attn_V, self.attn_U, self.attn_w = nn.Linear(d, attn_dim), nn.Linear(d, attn_dim), nn.Linear(attn_dim, 1)

    # --- giữ nguyên giao diện của GatedDetector (train/evaluate/demo dùng chung)
    spec = property(lambda self: self.base.spec)
    sp_groups = property(lambda self: self.base.sp_groups)
    vec_groups = property(lambda self: self.base.vec_groups)
    use_gates = property(lambda self: self.base.use_gates)
    needs_clip = property(lambda self: self.base.needs_clip)
    gate = property(lambda self: self.base.gate)

    def gates(self):
        return self.base.gates()

    def gate_dict(self):
        return self.base.gate_dict()

    def tile_embed(self, x, clip_emb=None):
        return self.base.embed(x, clip_emb)

    def tile_logits(self, h):
        """Điểm riêng của từng mảnh (cùng lớp phân loại) — dùng để hiển thị, không dùng để quyết định."""
        return self.base.head(h.float()).squeeze(1)

    def pool(self, h, counts=None):
        """h: [tổng số mảnh, D]; counts: [số ảnh]. Trả về (logit mỗi ảnh, trọng số chú ý mỗi mảnh)."""
        h = h.float()
        if counts is None:
            counts = torch.ones(len(h), dtype=torch.long, device=h.device)
        counts = counts.to(h.device)
        n = len(counts)
        seg = torch.repeat_interleave(torch.arange(n, device=h.device), counts)
        s = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(1)
        a = segment_softmax(s, seg, n)
        z = torch.zeros(n, h.shape[1], device=h.device, dtype=h.dtype).index_add(0, seg, a[:, None] * h)
        return self.base.head(z).squeeze(1), a

    def forward(self, x, clip_emb=None, counts=None):
        return self.pool(self.tile_embed(x, clip_emb), counts)[0]


class PlainBackbone(nn.Module):
    """Mô hình timm pretrained trên ảnh RGB (resize về img_size). Không cổng."""
    needs_clip = False
    use_gates = False
    spec: list = []

    def __init__(self, name, pretrained=True, img_size=224, freeze=False):
        super().__init__()
        import timm
        try:
            self.m = timm.create_model(name, pretrained=pretrained, num_classes=1, img_size=img_size)
        except TypeError:
            self.m = timm.create_model(name, pretrained=pretrained, num_classes=1)
        cfg = getattr(self.m, "pretrained_cfg", {}) or {}
        self.img_size = img_size
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        if freeze:  # linear probe: chỉ train lớp phân loại
            head = self.m.get_classifier()
            for p in self.m.parameters():
                p.requires_grad_(False)
            for p in head.parameters():
                p.requires_grad_(True)
        self.gate = nn.Parameter(torch.zeros(0), requires_grad=False)

    def gate_dict(self):
        return {}

    def forward(self, x, clip_emb=None):
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return self.m((x - self.mean) / self.std).squeeze(1)


class ClipFinetune(nn.Module):
    """CLIP visual encoder, fine-tune n khối transformer cuối + đầu tuyến tính. n=0: linear probe (~UnivFD)."""
    needs_clip = False
    use_gates = False
    spec: list = []
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, arch="ViT-L-14", pretrained="openai", n_blocks=2):
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.visual = model.visual
        for p in self.visual.parameters():
            p.requires_grad_(False)
        blocks = getattr(getattr(self.visual, "transformer", None), "resblocks", [])
        for b in list(blocks)[len(blocks) - n_blocks:] if n_blocks > 0 else []:
            for p in b.parameters():
                p.requires_grad_(True)
        if n_blocks > 0:
            for name in ("ln_post",):
                if hasattr(self.visual, name):
                    for p in getattr(self.visual, name).parameters():
                        p.requires_grad_(True)
        self.head = nn.Linear(self.visual.output_dim, 1)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))
        self.gate = nn.Parameter(torch.zeros(0), requires_grad=False)

    def gate_dict(self):
        return {}

    def forward(self, x, clip_emb=None):
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
        e = self.visual((x - self.mean) / self.std).float()
        return self.head(e / e.norm(dim=-1, keepdim=True)).squeeze(1)


def window_positions(h: int, w: int, size: int, stride: int, max_n: int = 0) -> list[tuple[int, int]]:
    """Góc trên-trái các cửa sổ size x size trượt với bước stride, phủ kín lưới h x w (cửa sổ cuối sát mép)."""
    def axis(n):
        p = list(range(0, max(1, n - size + 1), stride))
        if not p or p[-1] != n - size:
            p.append(max(0, n - size))
        return p
    pos = [(y, x) for y in axis(h) for x in axis(w)]
    if max_n and len(pos) > max_n:  # giới hạn: lấy đều trên toàn ảnh
        pos = [pos[i] for i in np.linspace(0, len(pos) - 1, max_n).round().astype(int)]
    return pos


class FullImageMIL(TileMIL):
    """KHÔNG cắt ảnh trước khi trích đặc trưng. Ảnh kích thước bất kỳ:
    1. RGB / HH / WPT / NPR được tính trên CẢ ẢNH -> bản đồ [C, H/4, W/4]; qua Conv 1x1 + cổng, cộng thành 1 bản đồ.
    2. Cửa sổ 64x64 trên bản đồ (= vùng 256x256 pixel) TRƯỢT với bước stride (mặc định 128 px, chồng lấn 50%)
       -> mỗi cửa sổ qua ViT-S/4 -> vector 384.
    3. FFT (cửa sổ Hann) và CLIP tính trên CẢ ẢNH -> vector toàn cục 64, ghép vào mọi cửa sổ.
    4. Các cửa sổ của cùng ảnh được liên kết bằng lớp chú ý (TileMIL.pool) -> 1 logit cho cả ảnh."""
    is_full = True

    def __init__(self, base, attn_dim=128, stride=128, train_windows=4, fft_window=True):
        super().__init__(base, attn_dim)
        self.stride_px, self.train_windows, self.fft_window = int(stride), int(train_windows), bool(fft_window)

    def _global_vec(self, x, clip_emb):
        b = self.base
        if not b.vec_groups:
            return None
        g = b.gates()
        o, vec, fft = len(b.sp_groups), [], None
        for grp in b.vec_groups:
            if grp.name.startswith("fft_b"):
                if fft is None:
                    fft = fft_radial(x[None].float(), window=self.fft_window).split(16, dim=1)
                vec.append(fft[int(grp.name[5:])])
            elif grp.name == "clip":
                if clip_emb is None:
                    raise ValueError("Cần embedding CLIP toàn ảnh")
                vec.append(clip_emb.reshape(1, -1).float())
        return sum(g[o + j] * b.projs[j](vec[j]) for j in range(len(vec)))  # [1, 64]

    def _fused_map_impl(self, x):
        b = self.base
        sp, _ = compute_groups(x[None], b.sp_groups)
        g = b.gates()
        return sum(g[i] * b.stems[i](sp[i]) for i in range(len(sp)))[0]  # [width, H/4, W/4]

    def _fused_map(self, x):
        # Khi train: KHÔNG giữ activation của compute_groups + 21 stem trên CẢ ẢNH cho backward (với 6 nhóm đặc trưng,
        # 8 ảnh/bước tốn ~0.5-1 GB -> tràn VRAM 4 GB sang RAM, chậm 20 lần). Tính lại ở backward (rẻ), kết quả y hệt.
        if self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._fused_map_impl, x, use_reentrant=False)
        return self._fused_map_impl(x)

    def embed_image(self, x, clip_emb=None, train=False, max_windows=0, chunk=64):
        """x: [3,H,W] (H,W bội của 4, >= 256). Trả về (h [n, D], vị trí pixel [(y,x)] của từng cửa sổ)."""
        b = self.base
        v = self._global_vec(x, clip_emb)
        if not b.sp_groups:
            return v, [(0, 0)]
        m = self._fused_map(x)
        win, st = 64, max(1, self.stride_px // 4)
        pos = window_positions(m.shape[1], m.shape[2], win, st, 0 if train else max_windows)
        if train and self.train_windows and len(pos) > self.train_windows:
            import random
            pos = random.sample(pos, self.train_windows)
        feats = []
        for s in range(0, len(pos), chunk):
            w = torch.stack([m[:, a:a + win, c:c + win] for a, c in pos[s:s + chunk]])
            feats.append(b.trunk(w))
        h = torch.cat(feats)
        if v is not None:
            h = torch.cat([h, v.expand(len(h), -1)], 1)
        return h, [(a * 4, c * 4) for a, c in pos]

    def forward_images(self, xs, clip_embs=None, train=False, max_windows=0, chunk=64):
        hs, counts = [], []
        for k, x in enumerate(xs):
            h, _ = self.embed_image(x, clip_embs[k] if clip_embs is not None else None, train, max_windows, chunk)
            hs.append(h); counts.append(len(h))
        return self.pool(torch.cat(hs), torch.tensor(counts, device=hs[0].device))[0]


# tên gợi nhớ cho các mô hình mạnh trong khảo sát
ALIASES = {
    "vit_rgb": "rgb:vit_small_patch16_224.augreg_in21k",
    "resnet50": "rgb:resnet50.a1_in1k",
    "convnext": "rgb:convnext_tiny.fb_in22k",
    "dinov2": "rgb:vit_small_patch14_dinov2.lvd142m",
}


def build_model(cfg_train: dict, spec: list[Group], use_gates: bool, clip_cfg: dict | None = None):
    t = ALIASES.get(cfg_train["trunk"], cfg_train["trunk"])
    if t.startswith("rgb:"):
        return PlainBackbone(t[4:], pretrained=cfg_train.get("pretrained", True),
                             img_size=cfg_train.get("img_size", 224), freeze=cfg_train.get("freeze_backbone", False))
    if t == "clip_ft":
        c = clip_cfg or {"arch": "ViT-L-14", "pretrained": "openai"}
        return ClipFinetune(c["arch"], c["pretrained"], cfg_train.get("clip_ft_blocks", 2))
    m = GatedDetector(spec, trunk=t, width=cfg_train["width"], pretrained=cfg_train.get("pretrained", False),
                      use_gates=use_gates, crop=cfg_train["crop"])
    mil = cfg_train.get("mil") or {}
    if not mil.get("enable"):
        return m
    if mil.get("feature_mode", "tile") == "full":
        return FullImageMIL(m, int(mil.get("attn_dim", 128)), int(mil.get("stride", 128)),
                            int(mil.get("windows_train", 4)), bool(mil.get("fft_window", True)))
    return TileMIL(m, int(mil.get("attn_dim", 128)))
