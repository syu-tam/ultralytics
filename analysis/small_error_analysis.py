import json
import math
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


SMALL_AREA_MAX = 32 * 32


def bbox_xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def is_small_ann(ann):
    return ann.get("area", ann["bbox"][2] * ann["bbox"][3]) < SMALL_AREA_MAX


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def compute_ap_from_pr(tp_flags, fp_flags, num_gt):
    """
    tp_flags, fp_flags: score 降順に並んだ detection の 0/1 配列
    VOC/COCO 風の積分 AP を簡易実装
    """
    if num_gt == 0:
        return 0.0, np.array([]), np.array([])

    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)

    recalls = tp_cum / num_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap), precisions, recalls


def prepare_small_subset(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)

    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # small GT の annotation id / image id を収集
    small_ann_ids = []
    small_img_ids = set()
    small_gt_by_image_cat = defaultdict(list)

    for ann in coco_gt.dataset["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if is_small_ann(ann):
            small_ann_ids.append(ann["id"])
            small_img_ids.add(ann["image_id"])
            small_gt_by_image_cat[(ann["image_id"], ann["category_id"])].append(ann)

    # small GT が存在する画像に限定した prediction を収集
    preds_on_small_images = [p for p in preds if p["image_id"] in small_img_ids]

    return coco_gt, preds, preds_on_small_images, small_gt_by_image_cat, small_img_ids


def eval_small_coco(gt_json_path, pred_json_path):
    """
    pycocotools の official evaluation で AP_S / AR_S を取る
    """
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.areaRng = [
        [0**2, 1e5**2],
        [0**2, 32**2],
        [32**2, 96**2],
        [96**2, 1e5**2],
    ]
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    # COCOeval.stats の一般的な意味:
    # 0: AP@[.50:.95]
    # 1: AP50
    # 2: AP75
    # 3: AP small
    # 4: AP medium
    # 5: AP large
    # 6: AR@1
    # 7: AR@10
    # 8: AR@100
    # 9: AR small
    # 10: AR medium
    # 11: AR large
    return {
        "AP_all": float(stats[0]),
        "AP50_all": float(stats[1]),
        "AP75_all": float(stats[2]),
        "AP_small": float(stats[3]),
        "AR_small": float(stats[9]),
    }


def greedy_match_small(gt_json_path, pred_json_path, iou_thr=0.5):
    """
    small GT のみを正解集合として、prediction を greedy matching する。
    matching は image_id, category_id ごと。
    これで small 用の TP/FP/FN, precision/recall を出す。
    """
    coco_gt, preds_all, preds_on_small_images, small_gt_by_image_cat, small_img_ids = prepare_small_subset(
        gt_json_path, pred_json_path
    )

    # prediction を image_id, category_id ごとに整理
    pred_by_image_cat = defaultdict(list)
    for p in preds_on_small_images:
        pred_by_image_cat[(p["image_id"], p["category_id"])].append(p)

    # score 降順
    for k in pred_by_image_cat:
        pred_by_image_cat[k].sort(key=lambda x: x["score"], reverse=True)

    all_scores = []
    all_tp = []
    all_fp = []

    image_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "scores": [], "tp_flags": [], "fp_flags": [], "num_gt": 0})

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    keys = set(list(small_gt_by_image_cat.keys()) + list(pred_by_image_cat.keys()))

    for key in keys:
        image_id, category_id = key
        gts = small_gt_by_image_cat.get(key, [])
        preds = pred_by_image_cat.get(key, [])

        gt_used = [False] * len(gts)
        total_gt += len(gts)
        class_stats[category_id]["num_gt"] += len(gts)

        gt_boxes = [bbox_xywh_to_xyxy(g["bbox"]) for g in gts]

        for pred in preds:
            pbox = bbox_xywh_to_xyxy(pred["bbox"])
            best_iou = -1.0
            best_j = -1

            for j, gt_box in enumerate(gt_boxes):
                if gt_used[j]:
                    continue
                iou = box_iou_xyxy(pbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            matched = best_j >= 0 and best_iou >= iou_thr
            score = float(pred["score"])

            all_scores.append(score)
            class_stats[category_id]["scores"].append(score)

            if matched:
                gt_used[best_j] = True
                all_tp.append(1)
                all_fp.append(0)
                class_stats[category_id]["tp_flags"].append(1)
                class_stats[category_id]["fp_flags"].append(0)

                total_tp += 1
                image_stats[image_id]["tp"] += 1
                class_stats[category_id]["tp"] += 1
            else:
                all_tp.append(0)
                all_fp.append(1)
                class_stats[category_id]["tp_flags"].append(0)
                class_stats[category_id]["fp_flags"].append(1)

                total_fp += 1
                image_stats[image_id]["fp"] += 1
                class_stats[category_id]["fp"] += 1

        fn_here = sum(1 for x in gt_used if not x)
        total_fn += fn_here
        image_stats[image_id]["fn"] += fn_here
        class_stats[category_id]["fn"] += fn_here

    # 全体 AP
    order = np.argsort(-np.array(all_scores)) if len(all_scores) > 0 else np.array([], dtype=int)
    all_tp_sorted = np.array(all_tp)[order] if len(order) > 0 else np.array([])
    all_fp_sorted = np.array(all_fp)[order] if len(order) > 0 else np.array([])

    ap, _, _ = compute_ap_from_pr(all_tp_sorted, all_fp_sorted, total_gt)

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # クラス別
    class_summary = {}
    for category_id, st in class_stats.items():
        scores = np.array(st["scores"])
        tp_flags = np.array(st["tp_flags"])
        fp_flags = np.array(st["fp_flags"])

        if len(scores) > 0:
            order = np.argsort(-scores)
            tp_sorted = tp_flags[order]
            fp_sorted = fp_flags[order]
        else:
            tp_sorted = np.array([])
            fp_sorted = np.array([])

        class_ap, _, _ = compute_ap_from_pr(tp_sorted, fp_sorted, st["num_gt"])
        class_precision = safe_div(st["tp"], st["tp"] + st["fp"])
        class_recall = safe_div(st["tp"], st["tp"] + st["fn"])
        class_f1 = safe_div(2 * class_precision * class_recall, class_precision + class_recall)

        class_summary[category_id] = {
            "num_gt_small": int(st["num_gt"]),
            "tp": int(st["tp"]),
            "fp": int(st["fp"]),
            "fn": int(st["fn"]),
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "ap": float(class_ap),
        }

    # 失敗画像ランキング
    worst_images = []
    for image_id, st in image_stats.items():
        worst_images.append({
            "image_id": image_id,
            "tp": int(st["tp"]),
            "fp": int(st["fp"]),
            "fn": int(st["fn"]),
            "error_score": int(st["fn"] * 2 + st["fp"]),
        })
    worst_images.sort(key=lambda x: (-x["error_score"], -x["fn"], -x["fp"], x["image_id"]))

    return {
        "iou_thr": iou_thr,
        "num_images_with_small_gt": int(len(small_img_ids)),
        "num_gt_small": int(total_gt),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap": float(ap),
        "class_summary": class_summary,
        "worst_images": worst_images[:50],
    }


def compare_two_models(gt_json_path, pred_a_path, pred_b_path, name_a="baseline", name_b="proposed"):
    # official COCO small metrics
    coco_a = eval_small_coco(gt_json_path, pred_a_path)
    coco_b = eval_small_coco(gt_json_path, pred_b_path)

    # greedy small metrics at IoU=0.50 and 0.75
    g50_a = greedy_match_small(gt_json_path, pred_a_path, iou_thr=0.50)
    g50_b = greedy_match_small(gt_json_path, pred_b_path, iou_thr=0.50)

    g75_a = greedy_match_small(gt_json_path, pred_a_path, iou_thr=0.75)
    g75_b = greedy_match_small(gt_json_path, pred_b_path, iou_thr=0.75)

    result = {
        name_a: {
            "coco": coco_a,
            "small_iou50": g50_a,
            "small_iou75": g75_a,
        },
        name_b: {
            "coco": coco_b,
            "small_iou50": g50_b,
            "small_iou75": g75_b,
        },
        "delta": {
            "AP_small": coco_b["AP_small"] - coco_a["AP_small"],
            "AR_small": coco_b["AR_small"] - coco_a["AR_small"],
            "P_small_iou50": g50_b["precision"] - g50_a["precision"],
            "R_small_iou50": g50_b["recall"] - g50_a["recall"],
            "F1_small_iou50": g50_b["f1"] - g50_a["f1"],
            "FN_small_iou50": g50_b["fn"] - g50_a["fn"],
            "FP_small_iou50": g50_b["fp"] - g50_a["fp"],
            "P_small_iou75": g75_b["precision"] - g75_a["precision"],
            "R_small_iou75": g75_b["recall"] - g75_a["recall"],
            "F1_small_iou75": g75_b["f1"] - g75_a["f1"],
            "FN_small_iou75": g75_b["fn"] - g75_a["fn"],
            "FP_small_iou75": g75_b["fp"] - g75_a["fp"],
        }
    }
    return result


def format_ratio(x):
    return f"{100.0 * x:.2f}"


def print_summary(comp, name_a, name_b):
    a = comp[name_a]
    b = comp[name_b]
    d = comp["delta"]

    print("\n=== COCO official small metrics ===")
    print(f"{name_a:>12s} | AP_S={format_ratio(a['coco']['AP_small'])} | AR_S={format_ratio(a['coco']['AR_small'])} | AP50={format_ratio(a['coco']['AP50_all'])} | AP75={format_ratio(a['coco']['AP75_all'])}")
    print(f"{name_b:>12s} | AP_S={format_ratio(b['coco']['AP_small'])} | AR_S={format_ratio(b['coco']['AR_small'])} | AP50={format_ratio(b['coco']['AP50_all'])} | AP75={format_ratio(b['coco']['AP75_all'])}")
    print(f"{'delta':>12s} | AP_S={(100.0*d['AP_small']):+.2f} | AR_S={(100.0*d['AR_small']):+.2f}")

    print("\n=== Small-only greedy metrics @ IoU=0.50 ===")
    print(f"{name_a:>12s} | GT={a['small_iou50']['num_gt_small']} | TP={a['small_iou50']['tp']} | FP={a['small_iou50']['fp']} | FN={a['small_iou50']['fn']} | P={format_ratio(a['small_iou50']['precision'])} | R={format_ratio(a['small_iou50']['recall'])} | F1={format_ratio(a['small_iou50']['f1'])}")
    print(f"{name_b:>12s} | GT={b['small_iou50']['num_gt_small']} | TP={b['small_iou50']['tp']} | FP={b['small_iou50']['fp']} | FN={b['small_iou50']['fn']} | P={format_ratio(b['small_iou50']['precision'])} | R={format_ratio(b['small_iou50']['recall'])} | F1={format_ratio(b['small_iou50']['f1'])}")
    print(f"{'delta':>12s} | FP={d['FP_small_iou50']:+d} | FN={d['FN_small_iou50']:+d} | P={(100.0*d['P_small_iou50']):+.2f} | R={(100.0*d['R_small_iou50']):+.2f} | F1={(100.0*d['F1_small_iou50']):+.2f}")

    print("\n=== Small-only greedy metrics @ IoU=0.75 ===")
    print(f"{name_a:>12s} | GT={a['small_iou75']['num_gt_small']} | TP={a['small_iou75']['tp']} | FP={a['small_iou75']['fp']} | FN={a['small_iou75']['fn']} | P={format_ratio(a['small_iou75']['precision'])} | R={format_ratio(a['small_iou75']['recall'])} | F1={format_ratio(a['small_iou75']['f1'])}")
    print(f"{name_b:>12s} | GT={b['small_iou75']['num_gt_small']} | TP={b['small_iou75']['tp']} | FP={b['small_iou75']['fp']} | FN={b['small_iou75']['fn']} | P={format_ratio(b['small_iou75']['precision'])} | R={format_ratio(b['small_iou75']['recall'])} | F1={format_ratio(b['small_iou75']['f1'])}")
    print(f"{'delta':>12s} | FP={d['FP_small_iou75']:+d} | FN={d['FN_small_iou75']:+d} | P={(100.0*d['P_small_iou75']):+.2f} | R={(100.0*d['R_small_iou75']):+.2f} | F1={(100.0*d['F1_small_iou75']):+.2f}")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="/workspace/datasets/coco/annotations/instances_val2017.json", help="COCO annotation json")
    parser.add_argument("--pred-a", required=True, help="baseline の prediction json")
    parser.add_argument("--pred-b", required=True, help="proposed の prediction json")
    parser.add_argument("--name-a", default="baseline")
    parser.add_argument("--name-b", default="proposed")
    parser.add_argument("--out", default="small_error_analysis.json")
    args = parser.parse_args()

    comp = compare_two_models(
        gt_json_path=args.gt,
        pred_a_path=args.pred_a,
        pred_b_path=args.pred_b,
        name_a=args.name_a,
        name_b=args.name_b,
    )

    print_summary(comp, args.name_a, args.name_b)
    save_json(comp, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()