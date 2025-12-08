import numpy as np


def yolo_norm_to_xyxy_norm(box):
    # box = [cx, cy, w, h] normalized (0..1)
    cx, cy, w, h = box
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return np.array([x1, y1, x2, y2])


def iou_xyxy_norm(a, b):
    # a, b: [x1,y1,x2,y2] normalized (0..1)
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-9) if union > 0 else 0.0


def evaluate_single_image_greedy(preds, gts, iou_thres=0.5):
    """
    preds: list of [cls, cx, cy, w, h, conf]  (normalized)
    gts:   list of [cls, cx, cy, w, h]        (normalized)
    returns: TP, FP, FN, matches_list
      matches_list: list of tuples (pred_idx, gt_idx, iou)
    """
    # convert to structured lists
    preds_struct = []
    for i, p in enumerate(preds):
        cls = int(p[0])
        conf = float(p[5])
        bbox_xyxy = yolo_norm_to_xyxy_norm(p[1:5])
        preds_struct.append(
            {"idx": i, "class": cls, "conf": conf, "bbox": bbox_xyxy})

    gts_struct = []
    for j, g in enumerate(gts):
        cls = int(g[0])
        bbox_xyxy = yolo_norm_to_xyxy_norm(g[1:5])
        gts_struct.append(
            {"idx": j, "class": cls, "bbox": bbox_xyxy, "used": False})

    # sort preds theo confidence giảm dần
    preds_struct.sort(key=lambda x: -x["conf"])

    TP = 0
    FP = 0
    matches = []

    for p in preds_struct:
        best_iou = 0.0
        best_gt = None
        for gt in gts_struct:
            if gt["used"]:
                continue
            if gt["class"] != p["class"]:
                continue
            iou_v = iou_xyxy_norm(p["bbox"], gt["bbox"])
            if iou_v > best_iou:
                best_iou = iou_v
                best_gt = gt

        if best_gt is not None and best_iou >= iou_thres:
            TP += 1
            best_gt["used"] = True
            matches.append((p["idx"], best_gt["idx"], best_iou))
        else:
            FP += 1
            matches.append((p["idx"], None, best_iou))  # no GT matched

    # FN = số GT chưa được dùng
    FN = sum(1 for g in gts_struct if not g["used"])

    return TP, FP, FN, matches


if __name__ == "__main__":
    preds = [
        [1, 0.277864, 0.332407, 0.348958, 0.448148, 0.715868],
        [1, 0.171615, 0.325926, 0.138542, 0.424074, 0.364280],
        [3, 0.578385, 0.333796, 0.100000, 0.263889, 0.203313]
    ]

    gts = [
        [1, 0.280000, 0.330000, 0.350000, 0.450000],
        [3, 0.580000, 0.334000, 0.095000, 0.260000]
    ]

    TP, FP, FN, matches = evaluate_single_image_greedy(
        preds, gts, iou_thres=0.5)
    print("TP, FP, FN:", TP, FP, FN)
    print("matches:", matches)
