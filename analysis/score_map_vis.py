#!/usr/bin/env python3
"""
Detect ヘッドのスコアマップ可視化・比較（YOLO11対応）

NMS前の生スコアを直接可視化するため、1回の推論で完結。
「モデルがどこに物体を見ているか」を P3/P4/P5 スケール別に表示。

単一モデル:
    python score_map_vis.py --models best.pt --source image.jpg

2モデル比較:
    python score_map_vis.py --models baseline_l.pt ours_l.pt --source image.jpg

クラス指定:
    python score_map_vis.py --models baseline_l.pt ours_l.pt --source image.jpg --cls 0

クラス一覧:
    python score_map_vis.py --models best.pt --list-classes

出力:
    - 各モデル単体の P3/P4/P5/combined
    - 複数モデル比較グリッド
    - 2モデル時は差分ヒートマップ
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


# ──────────────────────────────────────────────
# 前処理
# ──────────────────────────────────────────────

def letterbox(img: np.ndarray, size: int = 640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad = np.full((size, size, 3), 114, dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    pad[dh: dh + nh, dw: dw + nw] = resized
    return pad, scale, (dw, dh), (nw, nh)


def to_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = img_bgr[:, :, ::-1].copy()
    return (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


# ──────────────────────────────────────────────
# スコアマップ取得
# ──────────────────────────────────────────────

def _extract_main_preds(preds):
    """
    モデル出力から main 側の dict(boxes, scores, feats) を取り出す。

    想定パターン:
      - self-distill: {"main": {...}, "aux": {...}}
      - 直接 dict: {"boxes": ..., "scores": ..., "feats": ...}
      - list/tuple: その中に dict が入っている場合を探索
    """
    if isinstance(preds, dict):
        if "main" in preds:
            return preds["main"]
        return preds

    if isinstance(preds, (list, tuple)):
        for x in preds:
            if isinstance(x, dict):
                if "main" in x:
                    return x["main"]
                if "scores" in x and "feats" in x:
                    return x

    raise TypeError(
        f"Detect ヘッド出力の解釈に失敗しました。type(preds)={type(preds)}"
    )


def get_score_maps(det_model: torch.nn.Module, img_tensor: torch.Tensor) -> list:
    """
    Detect ヘッドの生クラススコア（sigmoid後）を各スケールで返す。

    Returns:
        list of (nc, Hi, Wi) float tensors  [P3, P4, P5 の順]
    """
    detect = det_model.model[-1]
    orig_training = detect.training
    detect.training = True

    try:
        with torch.no_grad():
            preds = det_model(img_tensor)
    finally:
        detect.training = orig_training

    preds = _extract_main_preds(preds)

    if not isinstance(preds, dict):
        raise TypeError(
            f"想定外の Detect 出力です。type(preds)={type(preds)}"
        )

    if "scores" not in preds or "feats" not in preds:
        raise KeyError(
            f"'scores' または 'feats' が見つかりません。利用可能キー: {list(preds.keys())}"
        )

    scores = preds["scores"].sigmoid()  # (1, nc, total_anchors)
    feats = preds["feats"]              # [P3_feat, P4_feat, P5_feat]

    if not isinstance(feats, (list, tuple)):
        raise TypeError(f"'feats' が list/tuple ではありません: {type(feats)}")

    splits = [f.shape[2] * f.shape[3] for f in feats]   # anchors per scale
    per_scale = torch.split(scores[0], splits, dim=-1)  # [(nc, Hi*Wi), ...]

    result = []
    for sc, feat in zip(per_scale, feats):
        h, w = feat.shape[2], feat.shape[3]
        result.append(sc.reshape(-1, h, w).cpu())       # (nc, H, W)
    return result


# ──────────────────────────────────────────────
# 可視化ユーティリティ
# ──────────────────────────────────────────────

def to_original_size(
    feat_map: np.ndarray,
    orig_hw,
    pad_dw_dh,
    resized_wh,
    input_size: int = 640,
) -> np.ndarray:
    """特徴マップサイズ → 元画像サイズへリマップ（パディング除去込み）"""
    full = cv2.resize(
        feat_map.astype(np.float32),
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    dw, dh = pad_dw_dh
    nw, nh = resized_wh
    cropped = full[dh: dh + nh, dw: dw + nw]
    oh, ow = orig_hw
    return cv2.resize(cropped, (ow, oh), interpolation=cv2.INTER_LINEAR)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8)


def overlay(img_bgr: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * heatmap_01), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)


def overlay_diverging(
    img_bgr: np.ndarray,
    signed_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    signed_map を [-1, 1] 想定で可視化する。
      - 正: 赤系（後段で ours > baseline を意味づけ可能）
      - 負: 青系
      - 0 : 白付近
    """
    x = np.clip(signed_map, -1.0, 1.0)
    vis = ((x + 1.0) * 127.5).astype(np.uint8)  # [-1,1] -> [0,255]
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_SEISMIC) if hasattr(cv2, "COLORMAP_SEISMIC") else cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def draw_detections(img: np.ndarray, result, names: dict) -> np.ndarray:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls)
        conf = float(box.conf)
        cls_name = names[cls_id] if cls_id in names else str(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls_name} {conf:.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return img


def make_grid(images: list, labels: list, cols: int = 2) -> np.ndarray:
    """画像リストを cols 列のグリッドにまとめる"""
    if len(images) == 0:
        raise ValueError("images が空です。")

    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, (img, lbl) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        cell = cv2.resize(img, (w, h))
        cell = put_label(cell, lbl)
        grid[r * h: (r + 1) * h, c * w: (c + 1) * w] = cell

    return grid


def safe_stem(path_str: str) -> str:
    return Path(path_str).stem.replace(" ", "_")


# ──────────────────────────────────────────────
# モデル1本分の処理
# ──────────────────────────────────────────────

def run_one_model(model_path: str, img_bgr: np.ndarray, args):
    from ultralytics import YOLO

    print(f"[*] Loading: {model_path}")
    yolo = YOLO(model_path)
    device = torch.device(args.device)
    det_model = yolo.model.to(device).eval()
    names = yolo.names

    if args.list_classes:
        print(f"\nClasses in {model_path}:")
        for i, n in names.items():
            print(f"  {i:3d}: {n}")
        return {
            "model_name": safe_stem(model_path),
            "names": names,
            "list_only": True,
        }

    img_input = img_bgr.copy()
    orig_hw = img_input.shape[:2]

    # 前処理
    padded, _, (dw, dh), (nw, nh) = letterbox(img_input, args.imgsz)
    img_t = to_tensor(padded, device)

    # スコアマップ取得
    print(f"[*] Extracting score maps from Detect head: {model_path}")
    scale_maps = get_score_maps(det_model, img_t)
    # [(nc, H3, W3), (nc, H4, W4), (nc, H5, W5)]

    if args.cls is not None:
        nc = int(scale_maps[0].shape[0])
        if not (0 <= args.cls < nc):
            raise ValueError(
                f"--cls={args.cls} は範囲外です。クラス数は {nc} です。"
            )

    heatmaps_orig = []
    scale_labels = ["P3", "P4", "P5"]
    strides = [args.imgsz // 8, args.imgsz // 16, args.imgsz // 32]

    for smap, slabel, stride in zip(scale_maps, scale_labels, strides):
        # (nc, H, W) -> (H, W)
        if args.cls is not None:
            spatial = smap[args.cls].numpy()
        else:
            spatial = smap.max(dim=0).values.numpy()

        heat_orig = to_original_size(
            normalize(spatial),
            orig_hw,
            (dw, dh),
            (nw, nh),
            args.imgsz,
        )
        heatmaps_orig.append(heat_orig)

        feat_h, feat_w = smap.shape[1], smap.shape[2]
        print(f"    - {slabel}: feat={feat_h}x{feat_w}, stride={stride}")

    combined_heat = normalize(np.stack(heatmaps_orig, axis=0).max(axis=0))

    # bbox 重畳版
    combined_vis = overlay(img_input.copy(), combined_heat, args.alpha)
    results = yolo.predict(img_input, conf=args.conf, verbose=False)
    draw_detections(combined_vis, results[0], names)

    # bbox なし版
    combined_vis_no_box = overlay(img_input.copy(), combined_heat, args.alpha)

    per_scale_vis = [overlay(img_input.copy(), h, args.alpha) for h in heatmaps_orig]

    return {
        "model_name": safe_stem(model_path),
        "names": names,
        "scale_labels": scale_labels,
        "heatmaps_orig": heatmaps_orig,          # [P3, P4, P5]
        "per_scale_vis": per_scale_vis,          # bboxなし
        "combined_heat": combined_heat,
        "combined_vis": combined_vis,            # bboxあり
        "combined_vis_no_box": combined_vis_no_box,
        "list_only": False,
    }


# ──────────────────────────────────────────────
# 保存処理
# ──────────────────────────────────────────────

def save_single_model_outputs(result: dict, img_bgr: np.ndarray, out_dir: Path, cls_tag: str):
    model_name = result["model_name"]

    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # P3/P4/P5
    for vis, slabel in zip(result["per_scale_vis"], result["scale_labels"]):
        path = model_dir / f"{slabel.lower()}_{cls_tag}.jpg"
        cv2.imwrite(str(path), vis)
        print(f"[*] Saved: {path}")

    # combined
    path_comb = model_dir / f"combined_{cls_tag}.jpg"
    cv2.imwrite(str(path_comb), result["combined_vis"])
    print(f"[*] Saved: {path_comb}")

    path_comb_nobox = model_dir / f"combined_nobox_{cls_tag}.jpg"
    cv2.imwrite(str(path_comb_nobox), result["combined_vis_no_box"])
    print(f"[*] Saved: {path_comb_nobox}")

    # 2x2 grid
    grid_imgs = result["per_scale_vis"] + [result["combined_vis"]]
    grid_labels = ["P3 small", "P4 medium", "P5 large", f"combined ({cls_tag})"]
    grid = make_grid(grid_imgs, grid_labels, cols=2)
    grid_path = model_dir / f"grid_{cls_tag}.jpg"
    cv2.imwrite(str(grid_path), grid)
    print(f"[*] Saved: {grid_path}")


def save_comparison_outputs(results_all: list, img_bgr: np.ndarray, out_dir: Path, cls_tag: str, alpha: float):
    cmp_dir = out_dir / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    scale_names = ["P3", "P4", "P5"]

    # スケール別比較
    for i, sname in enumerate(scale_names):
        imgs = []
        labels = []
        for res in results_all:
            imgs.append(overlay(img_bgr.copy(), res["heatmaps_orig"][i], alpha))
            labels.append(f"{res['model_name']} {sname}")
        grid = make_grid(imgs, labels, cols=len(imgs))
        path = cmp_dir / f"compare_{sname.lower()}_{cls_tag}.jpg"
        cv2.imwrite(str(path), grid)
        print(f"[*] Saved: {path}")

    # combined 比較（bboxなし）
    imgs = [res["combined_vis_no_box"] for res in results_all]
    labels = [f"{res['model_name']} combined" for res in results_all]
    grid = make_grid(imgs, labels, cols=len(imgs))
    path = cmp_dir / f"compare_combined_nobox_{cls_tag}.jpg"
    cv2.imwrite(str(path), grid)
    print(f"[*] Saved: {path}")

    # combined 比較（bboxあり）
    imgs = [res["combined_vis"] for res in results_all]
    labels = [f"{res['model_name']} combined bbox" for res in results_all]
    grid = make_grid(imgs, labels, cols=len(imgs))
    path = cmp_dir / f"compare_combined_bbox_{cls_tag}.jpg"
    cv2.imwrite(str(path), grid)
    print(f"[*] Saved: {path}")

    # 全体まとめグリッド
    summary_imgs = []
    summary_labels = []
    for res in results_all:
        summary_imgs.extend(res["per_scale_vis"] + [res["combined_vis"]])
        summary_labels.extend([
            f"{res['model_name']} P3",
            f"{res['model_name']} P4",
            f"{res['model_name']} P5",
            f"{res['model_name']} combined",
        ])
    summary_grid = make_grid(summary_imgs, summary_labels, cols=4)
    path = cmp_dir / f"summary_all_{cls_tag}.jpg"
    cv2.imwrite(str(path), summary_grid)
    print(f"[*] Saved: {path}")


def save_difference_outputs(res_a: dict, res_b: dict, img_bgr: np.ndarray, out_dir: Path, cls_tag: str, alpha: float):
    """
    res_b - res_a を差分として保存する。
    例: baseline -> ours の順で渡せば、正側が ours > baseline を意味する。
    """
    diff_dir = out_dir / "difference"
    diff_dir.mkdir(parents=True, exist_ok=True)

    scale_names = ["P3", "P4", "P5"]

    # 各スケールの絶対差
    for i, sname in enumerate(scale_names):
        a = res_a["heatmaps_orig"][i]
        b = res_b["heatmaps_orig"][i]
        abs_diff = normalize(np.abs(b - a))
        vis = overlay(img_bgr.copy(), abs_diff, alpha)
        vis = put_label(vis, f"abs diff: {res_b['model_name']} - {res_a['model_name']} ({sname})")
        path = diff_dir / f"absdiff_{sname.lower()}_{cls_tag}.jpg"
        cv2.imwrite(str(path), vis)
        print(f"[*] Saved: {path}")

    # combined 絶対差
    a = res_a["combined_heat"]
    b = res_b["combined_heat"]
    abs_diff = normalize(np.abs(b - a))
    abs_vis = overlay(img_bgr.copy(), abs_diff, alpha)
    abs_vis = put_label(abs_vis, f"abs diff: {res_b['model_name']} - {res_a['model_name']}")
    path = diff_dir / f"absdiff_combined_{cls_tag}.jpg"
    cv2.imwrite(str(path), abs_vis)
    print(f"[*] Saved: {path}")

    # combined signed diff
    signed = b.astype(np.float32) - a.astype(np.float32)
    denom = float(np.max(np.abs(signed))) + 1e-8
    signed_norm = signed / denom
    signed_vis = overlay_diverging(img_bgr.copy(), signed_norm, alpha)
    signed_vis = put_label(
        signed_vis,
        f"signed diff: {res_b['model_name']} - {res_a['model_name']}"
    )
    path = diff_dir / f"signeddiff_combined_{cls_tag}.jpg"
    cv2.imwrite(str(path), signed_vis)
    print(f"[*] Saved: {path}")

    # 差分まとめ
    grid = make_grid(
        [
            overlay(img_bgr.copy(), normalize(np.abs(res_b["heatmaps_orig"][0] - res_a["heatmaps_orig"][0])), alpha),
            overlay(img_bgr.copy(), normalize(np.abs(res_b["heatmaps_orig"][1] - res_a["heatmaps_orig"][1])), alpha),
            overlay(img_bgr.copy(), normalize(np.abs(res_b["heatmaps_orig"][2] - res_a["heatmaps_orig"][2])), alpha),
            signed_vis,
        ],
        [
            "abs diff P3",
            "abs diff P4",
            "abs diff P5",
            "signed diff combined",
        ],
        cols=2,
    )
    path = diff_dir / f"diff_grid_{cls_tag}.jpg"
    cv2.imwrite(str(path), grid)
    print(f"[*] Saved: {path}")


# ──────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="YOLO Detect head score map visualization / comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="YOLO .pt ファイルを1個以上指定",
    )
    ap.add_argument("--source", help="入力画像")
    ap.add_argument("--output", default="score_map_out", help="出力ディレクトリ")
    ap.add_argument("--cls", type=int, default=None, help="特定クラスIDのみ（省略時は全クラスmax）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="検出ボックス表示の閾値")
    ap.add_argument("--alpha", type=float, default=0.5, help="ヒートマップ不透明度")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--list-classes", action="store_true", help="クラス一覧を表示して終了")
    args = ap.parse_args()

    if not args.list_classes and not args.source:
        raise ValueError("--source が必要です。--list-classes の場合のみ省略可能です。")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # クラス一覧のみ
    if args.list_classes:
        for model_path in args.models:
            run_one_model(model_path, np.zeros((640, 640, 3), dtype=np.uint8), args)
        return

    img_bgr = cv2.imread(args.source)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {args.source}")

    results_all = []
    for model_path in args.models:
        res = run_one_model(model_path, img_bgr, args)
        if not res.get("list_only", False):
            results_all.append(res)

    if len(results_all) == 0:
        raise RuntimeError("有効な結果が得られませんでした。")

    # クラスタグ
    if args.cls is not None:
        # 1本目の names を代表として使う
        names0 = results_all[0]["names"]
        cls_name = names0.get(args.cls, "?")
        cls_tag = f"cls{args.cls}_{cls_name}"
    else:
        cls_tag = "max_all"

    # 各モデル単体出力
    for res in results_all:
        save_single_model_outputs(res, img_bgr, out_dir, cls_tag)

    # 複数モデル比較
    if len(results_all) >= 2:
        save_comparison_outputs(results_all, img_bgr, out_dir, cls_tag, args.alpha)

    # 2モデルのときは差分も出す
    if len(results_all) == 2:
        # 先に指定した順に baseline, ours を渡せば、
        # signed diff = ours - baseline になる
        save_difference_outputs(results_all[0], results_all[1], img_bgr, out_dir, cls_tag, args.alpha)

    print(f"\n[*] Filter: {cls_tag}")
    print(f"[*] Outputs: {out_dir}/")


if __name__ == "__main__":
    main()