# Val.py - ValidationManager with COCO-style mAP + curves & confusion matrix

import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


class ValidationManager:
    def __init__(self, iou_thres=0.5):
        """
        iou_thres: ngưỡng IoU để tính TP/FP/FN frame-level, P/R/F1.
        """
        self.iou_thres = iou_thres

        # Lưu để vẽ / thống kê per-frame
        self.saved_frames = []
        self.frames = []              # per-frame TP/FP/FN

        # Toàn bộ prediction / GT để tính mAP
        # prediction: {"img_id","cls","conf","box"}  (box: np.array[4] xyxy pixel)
        self.all_predictions = []
        # GT per image: img_id -> list of {"cls","box"}
        self.gt_by_image = {}

        # Tổng GT per class
        self.total_gt_per_class = {}  # cls -> count

    # -------------------------
    # IoU calculation (xyxy)
    # -------------------------
    @staticmethod
    def compute_iou(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = a1 + a2 - inter + 1e-9
        return inter / union

    # -------------------------
    # YOLO normalized pred -> pixel xyxy
    # pred: [cls, cx, cy, w, h, conf]
    # orig_shape: (H, W)
    # -------------------------
    def scale_pred_to_pixel(self, p, orig_shape):
        H, W = orig_shape
        cls = int(p[0])
        cx = p[1] * W
        cy = p[2] * H
        bw = p[3] * W
        bh = p[4] * H
        conf = float(p[5])
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        return cls, conf, np.array([x1, y1, x2, y2], dtype=float)

    # -------------------------
    # Load GT YOLO txt -> pixel xyxy
    # -------------------------
    def load_gt_from_txt_pixel(self, img_name, labels_folder, orig_shape):
        H, W = orig_shape
        base = os.path.splitext(img_name)[0]
        txt_path = os.path.join(labels_folder, base + ".txt")
        if not os.path.exists(txt_path):
            return []
        out = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx = float(parts[1]) * W
                cy = float(parts[2]) * H
                bw = float(parts[3]) * W
                bh = float(parts[4]) * H
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0
                out.append((cls, np.array([x1, y1, x2, y2], dtype=float)))
        return out

    # -------------------------
    # Process 1 frame
    # preds: list of [cls, cx, cy, w, h, conf] (normalized)
    # orig_shape: (H, W)
    # img_name: filename.jpg
    # -------------------------
    def process_frame(self, preds, orig_shape, img_name,
                      labels_folder='dataset/valid/labels'):
        H, W = orig_shape

        # scale preds
        scaled_preds = [self.scale_pred_to_pixel(p, orig_shape) for p in preds]

        # load GTs
        gts = self.load_gt_from_txt_pixel(img_name, labels_folder, orig_shape)

        # Lưu GT
        if img_name not in self.gt_by_image:
            self.gt_by_image[img_name] = []
        for gt_cls, gt_box in gts:
            self.gt_by_image[img_name].append({"cls": gt_cls, "box": gt_box})
            self.total_gt_per_class[gt_cls] = self.total_gt_per_class.get(
                gt_cls, 0) + 1

        # Lưu preds
        for cls, conf, box in scaled_preds:
            self.all_predictions.append(
                {"img_id": img_name, "cls": cls, "conf": conf, "box": box}
            )

        # ------------------ Frame-level TP/FP/FN ------------------
        used_gt_idx = set()
        TP = 0
        FP = 0

        preds_sorted = sorted(scaled_preds, key=lambda x: -x[1])

        for cls, conf, pbox in preds_sorted:
            best_iou = 0.0
            best_idx = -1
            for idx, (g_cls, g_box) in enumerate(gts):
                if g_cls != cls or idx in used_gt_idx:
                    continue
                iou = self.compute_iou(pbox, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= self.iou_thres and best_idx >= 0:
                TP += 1
                used_gt_idx.add(best_idx)
            else:
                FP += 1

        FN = len(gts) - len(used_gt_idx)
        self.frames.append(
            {"img_name": img_name, "TP": TP, "FP": FP, "FN": FN}
        )

        # lưu cho vẽ
        self.saved_frames.append(
            {"img_name": img_name, "orig_shape": orig_shape, "preds": scaled_preds}
        )

    # -------------------------
    # AP for one class @ a given IoU threshold (COCO-style 101-pt)
    # -------------------------
    def compute_ap_for_class_at_iou(self, cls, iou_thres):
        # Lấy GT per-image cho class này
        gt_per_image = {}
        for img, glist in self.gt_by_image.items():
            boxes = [g["box"] for g in glist if g["cls"] == cls]
            if boxes:
                gt_per_image[img] = boxes

        npos = sum(len(v) for v in gt_per_image.values())
        if npos == 0:
            return 0.0

        # Lọc prediction theo class, sort theo conf giảm dần
        preds = [p for p in self.all_predictions if p["cls"] == cls]
        preds = sorted(preds, key=lambda x: -x["conf"])

        # used flags cho từng GT (per-image)
        used = {img: np.zeros(len(boxes), dtype=bool)
                for img, boxes in gt_per_image.items()}

        tp = []
        fp = []

        for p in preds:
            img = p["img_id"]
            pbox = p["box"]

            if img not in gt_per_image:
                tp.append(0)
                fp.append(1)
                continue

            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(gt_per_image[img]):
                if used[img][idx]:
                    continue
                iou = self.compute_iou(pbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thres and best_idx >= 0:
                tp.append(1)
                fp.append(0)
                used[img][best_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        tp = np.array(tp, dtype=float)
        fp = np.array(fp, dtype=float)

        if tp.size == 0:
            return 0.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / (npos + 1e-9)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

        # precision envelope (monotone non-increasing)
        for i in range(len(precisions) - 2, -1, -1):
            if precisions[i] < precisions[i + 1]:
                precisions[i] = precisions[i + 1]

        # 101-point interpolation (COCO)
        recall_levels = np.linspace(0.0, 1.0, 101)
        precisions_interp = np.zeros_like(recall_levels)
        for i, r in enumerate(recall_levels):
            mask = recalls >= r
            if np.any(mask):
                precisions_interp[i] = np.max(precisions[mask])
            else:
                precisions_interp[i] = 0.0

        AP = np.mean(precisions_interp)
        return float(AP)

    # -------------------------
    # COCO-style mAP50, mAP50-95
    # -------------------------
    def compute_map_coco_style(self):
        classes = sorted(self.total_gt_per_class.keys())
        if not classes:
            return 0.0, 0.0, {}

        aps50 = []
        aps_per_class_all_iou = {}  # cls -> list of AP at each IoU

        iou_list = np.linspace(0.5, 0.95, 10)

        for c in classes:
            ap_list_c = []
            for t in iou_list:
                ap_t = self.compute_ap_for_class_at_iou(c, t)
                ap_list_c.append(ap_t)
            aps_per_class_all_iou[c] = ap_list_c
            aps50.append(ap_list_c[0])               # IoU=0.5 là index 0

        mAP50 = float(np.mean(aps50)) if aps50 else 0.0

        # mAP50-95 = mean over IoU=0.50:0.95 and over classes
        mean_per_class = [np.mean(aps_per_class_all_iou[c]) for c in classes]
        mAP5095 = float(np.mean(mean_per_class)) if mean_per_class else 0.0

        return mAP50, mAP5095, aps_per_class_all_iou

    # -------------------------
    # Build PR / P-R-F1 curves for một class
    # -------------------------
    def build_pr_curves(self, cls, iou_thres=0.5):
        # Lấy toàn bộ prediction của class này
        preds = [p for p in self.all_predictions if p["cls"] == cls]
        preds = sorted(preds, key=lambda x: -x["conf"])

        # Lấy nhãn thực
        gt_per_image_cls = {}
        for img, glist in self.gt_by_image.items():
            boxes = [g["box"] for g in glist if g["cls"] == cls]
            if boxes:
                gt_per_image_cls[img] = boxes

        npos = sum(len(v) for v in gt_per_image_cls.values())
        if npos == 0:
            return None

        used = {img: np.zeros(len(boxes), dtype=int)
                for img, boxes in gt_per_image_cls.items()}

        tp = []
        fp = []

        for p in preds:
            img = p["img_id"]
            pbox = p["box"]

            if img not in gt_per_image_cls:
                tp.append(0)
                fp.append(1)
                continue

            best_iou = 0.0
            best_idx = -1

            for idx, gt_box in enumerate(gt_per_image_cls[img]):
                if used[img][idx]:
                    continue
                i = self.compute_iou(pbox, gt_box)
                if i > best_iou:
                    best_iou = i
                    best_idx = idx

            if best_iou >= iou_thres and best_idx >= 0:
                tp.append(1)
                fp.append(0)
                used[img][best_idx] = 1
            else:
                tp.append(0)
                fp.append(1)

        tp = np.array(tp, dtype=float)
        fp = np.array(fp, dtype=float)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / (npos + 1e-9)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-9)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)

        return {
            "recall": recalls,
            "precision": precisions,
            "f1": f1_scores
        }

    # -------------------------
    # Confusion matrix (tất cả classes)
    # -------------------------
    def build_confusion_matrix(self, save_dir="runs/val_splited_model"):
        os.makedirs(save_dir, exist_ok=True)

        classes = sorted(self.total_gt_per_class.keys())
        nC = len(classes)
        if nC == 0:
            print("No classes for confusion matrix.")
            return

        cls_to_idx = {c: i for i, c in enumerate(classes)}

        # cm[pred][gt]: rows = Predicted, cols = Ground Truth
        cm = np.zeros((nC, nC), dtype=int)

        # Với mỗi image, duyệt GT và predictions
        for img, gts in self.gt_by_image.items():
            preds = [p for p in self.all_predictions if p["img_id"] == img]
            used_preds = set()

            for gt_obj in gts:
                gt_cls = gt_obj["cls"]
                gt_box = gt_obj["box"]

                best_iou = 0.0
                best_pred = -1
                best_pred_cls = None

                for i, p in enumerate(preds):
                    if i in used_preds:
                        continue
                    iou = self.compute_iou(p["box"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = i
                        best_pred_cls = p["cls"]

                if best_iou >= self.iou_thres and best_pred >= 0:
                    pred_idx = cls_to_idx[best_pred_cls]
                    gt_idx = cls_to_idx[gt_cls]
                    cm[pred_idx, gt_idx] += 1
                    used_preds.add(best_pred)
                else:
                    # FN (miss) không cộng vào matrix; bạn đã track FN riêng rồi
                    pass

        # Normalized confusion matrix theo GT (chuẩn hóa theo từng cột)
        cm_norm = cm.astype(float)
        for j in range(nC):
            col_sum = cm[:, j].sum()
            cm_norm[:, j] = cm[:, j] / (col_sum + 1e-9)

        # ----- Confusion matrix (counts) -----
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,  # Ground Truth (x-axis)
            yticklabels=classes,  # Predicted   (y-axis)
        )
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted")
        plt.title("Confusion Matrix (Counts)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()

        # ----- Confusion matrix (normalized by GT) -----
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted")
        plt.title("Confusion Matrix (Normalized by GT)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix_normalized.png"))
        plt.close()

        print(f"Saved confusion matrices to {save_dir}")

    # -------------------------
    # Vẽ P/R/F1-curve, PR-curve cho tất cả classes
    # -------------------------
    def build_all_curves(self, save_dir="runs/val_splited_model"):
        os.makedirs(save_dir, exist_ok=True)

        classes = sorted(self.total_gt_per_class.keys())
        if not classes:
            print("No classes for curves.")
            return

        # Lưu curves cho từng class
        per_class_curves = {}   # cls -> {"recall","precision","f1"}
        for c in classes:
            curves = self.build_pr_curves(c, self.iou_thres)
            if curves is None:
                continue
            per_class_curves[c] = curves

        if not per_class_curves:
            print("No valid curves to plot.")
            return

        # =========================================================
        # 1) PR-CURVE (Recall vs Precision) như Ultralytics
        #    - Mỗi class 1 đường
        #    - 1 đường "All classes (mean)" trên trục Recall chung [0..1]
        #    Lưu: PR_curve_ALL.png
        # =========================================================
        recall_grid = np.linspace(0.0, 1.0, 101)  # grid recall chung
        interp_precisions = []                    # [n_cls, 101]

        plt.figure(figsize=(7, 6))
        for c, curves in per_class_curves.items():
            r = curves["recall"]
            p = curves["precision"]

            if len(r) < 2:
                continue

            # sort theo recall tăng dần
            order = np.argsort(r)
            r_sorted = r[order]
            p_sorted = p[order]

            # bỏ trùng recall
            unique_r, idx = np.unique(r_sorted, return_index=True)
            unique_p = p_sorted[idx]

            # nội suy precision trên recall_grid
            p_interp = np.interp(
                recall_grid, unique_r, unique_p, left=0.0, right=unique_p[-1]
            )
            interp_precisions.append(p_interp)

            # vẽ đường class
            plt.plot(r_sorted, p_sorted, label=f"Class {c}", alpha=0.5)

        if interp_precisions:
            interp_precisions = np.stack(
                interp_precisions, axis=0)  # [n_cls, 101]
            mean_precision = interp_precisions.mean(axis=0)

            # vẽ đường "All classes" đậm màu
            plt.plot(
                recall_grid,
                mean_precision,
                label="All classes (mean)",
                color="black",
                linewidth=2.0,
            )

        plt.title("Precision-Recall Curve (All Classes)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "PR_curve_ALL.png"))
        plt.close()

        # =========================================================
        # 2) P / R / F1 vs index (gần giống Ultralytics)
        #    - Chuẩn hóa mỗi class về cùng số điểm N_common bằng nội suy
        #    - Vẽ:
        #        P_curve_ALL.png, R_curve_ALL.png, F1_curve_ALL.png
        #      mỗi file: nhiều đường class + 1 đường All classes (mean)
        # =========================================================
        N_common = 100  # số điểm chuẩn hóa (có thể đổi 50/101 nếu muốn)

        # collect P/R/F1 đã nội suy cho từng class
        P_interp_list = []
        R_interp_list = []
        F1_interp_list = []
        cls_list = []  # để map index -> class label

        for c, curves in per_class_curves.items():
            p = curves["precision"]
            r = curves["recall"]
            f1 = curves["f1"]

            n = len(p)
            if n < 2:
                continue

            x_orig = np.linspace(0.0, 1.0, n)
            x_new = np.linspace(0.0, 1.0, N_common)

            p_i = np.interp(x_new, x_orig, p)
            r_i = np.interp(x_new, x_orig, r)
            f1_i = np.interp(x_new, x_orig, f1)

            P_interp_list.append(p_i)
            R_interp_list.append(r_i)
            F1_interp_list.append(f1_i)
            cls_list.append(c)

        if not P_interp_list:
            print("Not enough data to interpolate P/R/F1 curves.")
            return

        P_interp = np.stack(P_interp_list, axis=0)   # [n_cls, N_common]
        R_interp = np.stack(R_interp_list, axis=0)
        F1_interp = np.stack(F1_interp_list, axis=0)

        P_mean = P_interp.mean(axis=0)
        R_mean = R_interp.mean(axis=0)
        F1_mean = F1_interp.mean(axis=0)

        x_idx = np.arange(N_common)

        # --------- Precision curve (P_curve_ALL.png) ---------
        plt.figure(figsize=(7, 6))
        for i, c in enumerate(cls_list):
            plt.plot(
                x_idx,
                P_interp[i],
                label=f"Class {c}",
                alpha=0.5,
            )
        plt.plot(
            x_idx,
            P_mean,
            label="All classes (mean)",
            color="black",
            linewidth=2.0,
        )
        plt.title("Precision Curve (All Classes)")
        plt.xlabel("Normalized Prediction Index")
        plt.ylabel("Precision")
        plt.ylim(0.0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "P_curve_ALL.png"))
        plt.close()

        # --------- Recall curve (R_curve_ALL.png) ---------
        plt.figure(figsize=(7, 6))
        for i, c in enumerate(cls_list):
            plt.plot(
                x_idx,
                R_interp[i],
                label=f"Class {c}",
                alpha=0.5,
            )
        plt.plot(
            x_idx,
            R_mean,
            label="All classes (mean)",
            color="black",
            linewidth=2.0,
        )
        plt.title("Recall Curve (All Classes)")
        plt.xlabel("Normalized Prediction Index")
        plt.ylabel("Recall")
        plt.ylim(0.0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "R_curve_ALL.png"))
        plt.close()

        # --------- F1 curve (F1_curve_ALL.png) ---------
        plt.figure(figsize=(7, 6))
        for i, c in enumerate(cls_list):
            plt.plot(
                x_idx,
                F1_interp[i],
                label=f"Class {c}",
                alpha=0.5,
            )
        plt.plot(
            x_idx,
            F1_mean,
            label="All classes (mean)",
            color="black",
            linewidth=2.0,
        )
        plt.title("F1 Curve (All Classes)")
        plt.xlabel("Normalized Prediction Index")
        plt.ylabel("F1-score")
        plt.ylim(0.0, 1.05)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "F1_curve_ALL.png"))
        plt.close()

        print(
            f"Saved PR_curve_ALL, P_curve_ALL, R_curve_ALL, F1_curve_ALL to {save_dir}")

    # -------------------------
    # Draw sample images với bbox
    # -------------------------

    def draw_samples(self, img_folder='dataset/valid/images',
                     save_folder='runs/val_splited_model'):
        os.makedirs(save_folder, exist_ok=True)
        if not self.saved_frames:
            print("No saved frames to draw.")
            return
        sample = random.sample(
            self.saved_frames, min(6, len(self.saved_frames)))
        for i, fr in enumerate(sample):
            img_path = os.path.join(img_folder, fr["img_name"])
            img = cv2.imread(img_path)
            if img is None:
                continue
            for cls, conf, box in fr["preds"]:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{cls}:{conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(save_folder, f"sample_{i+1}.jpg"), img)
        print(f"Saved {len(sample)} samples to {save_folder}")

    # -------------------------
    # Summary + vẽ biểu đồ
    # -------------------------
    def summary(self, img_folder='dataset/valid/images',
                save_dir="runs/val_splited_model"):
        os.makedirs(save_dir, exist_ok=True)

        total_TP = sum(f["TP"] for f in self.frames)
        total_FP = sum(f["FP"] for f in self.frames)
        total_FN = sum(f["FN"] for f in self.frames)

        print("\n=========== FINAL VALIDATION (Frame-level, IoU=%.2f) ===========" %
              self.iou_thres)
        print("Frames:", len(self.frames))
        print("TP:", total_TP)
        print("FP:", total_FP)
        print("FN:", total_FN)

        P = total_TP / (total_TP + total_FP + 1e-9)
        R = total_TP / (total_TP + total_FN + 1e-9)
        F1 = 2 * P * R / (P + R + 1e-9)

        print(f"Precision = {P:.4f}")
        print(f"Recall    = {R:.4f}")
        print(f"F1-score  = {F1:.4f}")

        # mAP COCO-style
        mAP50, mAP5095, aps_per_class = self.compute_map_coco_style()

        print("\n----- AP per class @ IoU=0.5 -----")
        classes = sorted(self.total_gt_per_class.keys())
        for c in classes:
            ap50_c = aps_per_class[c][0] if c in aps_per_class else 0.0
            print(f"AP50(Class {c}) = {ap50_c:.4f}")
        print(f"mAP50 = {mAP50:.4f}")

        print("\n----- mean AP per class over IoU=0.5:0.95 -----")
        for c in classes:
            mean_ap_c = np.mean(
                aps_per_class[c]) if c in aps_per_class else 0.0
            print(f"Class {c} mean AP = {mean_ap_c:.4f}")
        print(f"mAP50-95 = {mAP5095:.4f}")

        # Vẽ các biểu đồ
        self.draw_samples(img_folder=img_folder, save_folder=save_dir)
        self.build_all_curves(save_dir=save_dir)
        self.build_confusion_matrix(save_dir=save_dir)

        print("==========================================")
