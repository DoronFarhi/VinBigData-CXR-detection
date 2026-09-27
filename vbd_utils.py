"""Shared utilities for the VinBigData pipeline.

One source of truth for everything the notebooks share: paths, box operations,
the competition-style evaluation (VOC mAP at IoU 0.4), classifier/detector reports,
detector fusion, soft combine and submission formatting.

Conventions used everywhere:
- Boxes are normalized to [0, 1]: x / image width, y / image height.
- class_id 0..13 are findings, 14 is "No finding" (a full-image box 0 0 1 1).
- Prediction DataFrames: image_id, class_id, score, x_min, y_min, x_max, y_max.
- Ground-truth DataFrames: image_id, class_id, x_min, y_min, x_max, y_max, n_rads.
"""
import os
import glob
import fnmatch
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLASSES = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation',
           'ILD', 'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion',
           'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
NO_FINDING = 14
ALL_CLASSES = CLASSES + ['No finding']
IMG_SIZE = 1024
BOX = ['x_min', 'y_min', 'x_max', 'y_max']
PRED_COLS = ['image_id', 'class_id', 'score'] + BOX
# Pairs that radiologists label on the same region (EDA section 10)
PARTNER_PAIRS = [(10, 11), (4, 7)]   # Pleural effusion <-> Pleural thickening, Consolidation <-> Lung Opacity


# =============================================================================
# Paths
# =============================================================================
def competition_dir():
    """Locate the competition data (Kaggle has used two mount layouts)."""
    for d in ['/kaggle/input/vinbigdata-chest-xray-abnormalities-detection',
              '/kaggle/input/competitions/vinbigdata-chest-xray-abnormalities-detection']:
        if os.path.exists(f'{d}/train.csv'):
            return d
    raise FileNotFoundError('Competition data not found - add it to the notebook inputs.')


def find_dir(name, roots=('/kaggle/working', '/kaggle/input')):
    """Find a directory by name without descending into it (fast even with 18k PNGs inside)."""
    for root in roots:
        for cur, dirs, _ in os.walk(root):
            if name in dirs:
                return os.path.join(cur, name)
            dirs[:] = [d for d in dirs if d not in ('png1024', 'images', 'labels', 'train', 'test')]
    raise FileNotFoundError(f'Directory {name} not found - attach the notebook output that created it.')


def find_file(name, roots=('/kaggle/working', '/kaggle/input')):
    """Find a single file by exact name anywhere under the roots (first match wins)."""
    hits = find_all(name, roots)
    if not hits:
        raise FileNotFoundError(f'{name} not found - attach the notebook output that created it.')
    return hits[0]


def find_all(pattern, roots=('/kaggle/working', '/kaggle/input')):
    """All files matching a glob pattern (e.g. 'oof_yolo_fold*.csv'), de-duplicated by file name."""
    seen, out = set(), []
    skip = {'png1024', 'images', 'labels', 'train', 'test'}
    for root in roots:
        for cur, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in skip]
            for f in fnmatch.filter(files, pattern):
                if f not in seen:            # /kaggle/working wins over older copies in inputs
                    seen.add(f)
                    out.append(os.path.join(cur, f))
    return sorted(out)


def load_concat(pattern):
    """Concatenate all CSVs matching a pattern (e.g. all available OOF folds)."""
    files = find_all(pattern)
    if not files:
        raise FileNotFoundError(f'No files match {pattern}')
    print(f'{pattern}: {len(files)} file(s)')
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# =============================================================================
# Boxes
# =============================================================================
def iou_matrix(a, b):
    """Pairwise IoU between two arrays of [x_min, y_min, x_max, y_max] boxes."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms_per_class(df, iou_thr=0.5):
    """Greedy NMS inside each (image, class). Used only after partner-class copies."""
    keep = []
    for _, g in df.groupby(['image_id', 'class_id']):
        g = g.sort_values('score', ascending=False)
        boxes, idx = g[BOX].values, g.index.values
        alive = np.ones(len(g), bool)
        for i in range(len(g)):
            if not alive[i]:
                continue
            keep.append(idx[i])
            if i + 1 < len(g):
                ious = iou_matrix(boxes[i], boxes[i + 1:])[0]
                alive[i + 1:] &= ious < iou_thr
    return df.loc[keep].reset_index(drop=True)


# =============================================================================
# Ground truth
# =============================================================================
def build_gt(targets, image_ids):
    """Validation ground truth: fused boxes for abnormal images, a No finding box for normal ones."""
    ids = pd.Index(pd.unique(pd.Series(image_ids)))
    t = targets[targets.image_id.isin(ids)][['image_id', 'class_id'] + BOX + ['n_rads']]
    normal = ids.difference(t.image_id.unique())
    nf = pd.DataFrame({'image_id': normal, 'class_id': NO_FINDING,
                       'x_min': 0.0, 'y_min': 0.0, 'x_max': 1.0, 'y_max': 1.0, 'n_rads': 3})
    return pd.concat([t, nf], ignore_index=True)


# =============================================================================
# Evaluation: PASCAL VOC mAP (all-point interpolation), as in the competition
# =============================================================================
def voc_ap(rec, prec):
    """Area under the monotone precision envelope (VOC 2010+)."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match_class(gt_c, pr_c, iou_thr):
    """VOC matching for one class. Returns TP flags for preds (score-sorted) and the best
    matching score for every GT box (NaN = never detected)."""
    pr_c = pr_c.sort_values('score', ascending=False).reset_index(drop=True)
    tp = np.zeros(len(pr_c), bool)
    gt_score = np.full(len(gt_c), np.nan)
    gt_pos = gt_c.reset_index(drop=True).groupby('image_id').indices
    gt_boxes = gt_c[BOX].values
    for img, pidx in pr_c.groupby('image_id').indices.items():
        gidx = gt_pos.get(img)
        if gidx is None:
            continue                                          # every pred here is a FP
        ious = iou_matrix(pr_c.loc[pidx, BOX].values, gt_boxes[gidx])
        used = np.zeros(len(gidx), bool)
        for r, p in enumerate(pidx):                          # pidx is already in score order
            j = int(np.argmax(ious[r]))
            if ious[r, j] >= iou_thr and not used[j]:
                used[j] = True
                tp[p] = True
                gt_score[gidx[j]] = pr_c.score.iat[p]
    return pr_c, tp, gt_score


def evaluate(gt, preds, iou_thr=0.4, classes=None, verbose=True):
    """Competition-style evaluation.

    Returns a dict with:
      ap      - Series of AP per class name (NaN if the class has no GT in this subset)
      map15   - mean over all 15 classes (the competition metric)
      map14   - mean over the 14 findings (use this for detectors, which do not predict No finding)
      curves  - {class_id: (recall, precision, scores)} for PR plots
      gt      - copy of gt with 'det_score' (score of the matching prediction, NaN if missed)
    """
    classes = range(15) if classes is None else classes
    gt = gt.reset_index(drop=True).copy()
    gt['det_score'] = np.nan
    ap, curves = {}, {}
    for c in classes:
        gt_c = gt[gt.class_id == c]
        pr_c = preds[preds.class_id == c]
        if len(gt_c) == 0:
            ap[ALL_CLASSES[c]] = np.nan
            continue
        if len(pr_c) == 0:
            ap[ALL_CLASSES[c]] = 0.0
            curves[c] = (np.array([0.0]), np.array([0.0]), np.array([1.0]))
            continue
        pr_sorted, tp, gt_score = _match_class(gt_c, pr_c, iou_thr)
        ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
        rec = ctp / len(gt_c)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        ap[ALL_CLASSES[c]] = voc_ap(rec, prec)
        curves[c] = (rec, prec, pr_sorted.score.values)
        gt.loc[gt_c.index, 'det_score'] = gt_score
    ap = pd.Series(ap)
    findings = ap[[n for n in CLASSES if n in ap.index]]
    res = dict(ap=ap, map15=float(ap.mean()), map14=float(findings.mean()), curves=curves, gt=gt)
    if verbose:
        print(f'mAP@{iou_thr} - 14 findings: {res["map14"]:.4f} | all 15 classes: {res["map15"]:.4f}')
    return res


# =============================================================================
# Detector reports
# =============================================================================
def plot_ap_table(ap_by_stage, title='AP per class'):
    """ap_by_stage: {stage name: Series of AP per class}. Returns the table and draws a grouped bar chart."""
    tab = pd.DataFrame(ap_by_stage)
    ax = tab.plot.barh(figsize=(10, 0.45 * len(tab) + 1.5), width=0.8)
    ax.invert_yaxis(); ax.set_xlabel('AP @ IoU 0.4'); ax.set_title(title)
    ax.legend(loc='lower right'); plt.tight_layout(); plt.show()
    means = tab.loc[[c for c in CLASSES if c in tab.index]].mean().rename('mAP (14 findings)')
    return pd.concat([tab.round(3), means.to_frame().T.round(4)])


def plot_pr_curves(res, classes=range(14), title=''):
    """Precision-recall curve per class, AP in each panel title."""
    classes = [c for c in classes if c in res['curves']]
    n = len(classes); cols = 5; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.3 * rows))
    for ax, c in zip(np.ravel(axes), classes):
        rec, prec, _ = res['curves'][c]
        ax.plot(rec, prec, lw=1.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_title(f'{ALL_CLASSES[c]}\nAP={res["ap"][ALL_CLASSES[c]]:.3f}', fontsize=10)
        ax.set_xlabel('recall', fontsize=8); ax.set_ylabel('precision', fontsize=8)
    for ax in np.ravel(axes)[n:]:
        ax.axis('off')
    fig.suptitle(title); plt.tight_layout(); plt.show()


def detection_confusion(gt, preds, iou_thr=0.4, score_thr=0.2):
    """Confusion matrix for detection at a score threshold (findings only).

    Rows = GT class (+ 'Background' for predictions that hit no GT box).
    Cols = predicted class (+ 'Missed' for GT boxes with no overlapping prediction).
    A GT box counts as correct if any same-class prediction overlaps it; otherwise it is
    assigned to the best-overlapping prediction of another class, or 'Missed'.
    """
    gt = gt[gt.class_id != NO_FINDING]
    pr = preds[(preds.class_id != NO_FINDING) & (preds.score >= score_thr)]
    labels_r = CLASSES + ['Background']; labels_c = CLASSES + ['Missed']
    cm = pd.DataFrame(0, index=labels_r, columns=labels_c)
    pr_groups = pr.groupby('image_id')
    pr_hit = pd.Series(False, index=pr.index)
    for img, g in gt.groupby('image_id'):
        if img not in pr_groups.groups:
            for c in g.class_id:
                cm.loc[CLASSES[c], 'Missed'] += 1
            continue
        p = pr_groups.get_group(img)
        ious = iou_matrix(g[BOX].values, p[BOX].values)
        pr_hit.loc[p.index[(ious >= iou_thr).any(0)]] = True
        for r, c in enumerate(g.class_id.values):
            ok = ious[r] >= iou_thr
            if (ok & (p.class_id.values == c)).any():
                cm.loc[CLASSES[c], CLASSES[c]] += 1
            elif ok.any():
                j = int(np.argmax(np.where(ok, ious[r], -1)))
                cm.loc[CLASSES[c], CLASSES[p.class_id.values[j]]] += 1
            else:
                cm.loc[CLASSES[c], 'Missed'] += 1
    for c in pr.loc[~pr_hit, 'class_id']:
        cm.loc['Background', CLASSES[c]] += 1
    return cm


def plot_confusion(cm, title='', normalize=True):
    """Heatmap of a detection confusion matrix (rows normalized to sum to 1)."""
    data = cm.div(cm.sum(1).replace(0, 1), axis=0) if normalize else cm
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(data.values, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(data.shape[1])); ax.set_xticklabels(data.columns, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(data.shape[0])); ax.set_yticklabels(data.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if v >= 0.01:
                ax.text(j, i, f'{v:.2f}' if normalize else int(v), ha='center', va='center', fontsize=6.5,
                        color='white' if (normalize and v > 0.6) else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Ground truth'); ax.set_title(title)
    plt.colorbar(im, fraction=0.03); plt.tight_layout(); plt.show()


SIZE_BINS = [0, 16, 32, 64, 128, 256, 10_000]
SIZE_LABELS = ['<16', '16-32', '32-64', '64-128', '128-256', '>256']


def recall_breakdown(res, score_thr=0.0):
    """Recall of GT findings by box size (typical side in px at 1024) and by n_rads.
    score_thr=0 -> any detection counts; e.g. 0.2 -> only detections the model is confident about."""
    g = res['gt'][res['gt'].class_id != NO_FINDING].copy()
    g['hit'] = g.det_score.fillna(-1) >= max(score_thr, 1e-12)
    side = np.sqrt((g.x_max - g.x_min) * (g.y_max - g.y_min)) * IMG_SIZE
    g['size'] = pd.cut(side, SIZE_BINS, labels=SIZE_LABELS, right=False)
    g['class'] = g.class_id.map(dict(enumerate(CLASSES)))
    by_size = g.pivot_table(index='class', columns='size', values='hit', aggfunc='mean', observed=False)
    by_rads = g.pivot_table(index='class', columns='n_rads', values='hit', aggfunc='mean')
    counts = g.groupby(['size'], observed=False).size().rename('boxes')

    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    g.groupby('size', observed=False).hit.mean().plot.bar(ax=ax[0], color='#534AB7')
    ax[0].set_title(f'Recall by box size (score >= {score_thr})'); ax[0].set_ylim(0, 1); ax[0].set_xlabel('side at 1024 px')
    g.groupby('n_rads').hit.mean().plot.bar(ax=ax[1], color='#0F6E56')
    ax[1].set_title(f'Recall by radiologist agreement (score >= {score_thr})'); ax[1].set_ylim(0, 1)
    plt.tight_layout(); plt.show()
    return by_size.round(2), by_rads.round(2), counts


def show_predictions(image_id, gt, preds, png_dir, score_thr=0.2, ax=None, title=None):
    """Draw GT (green, dashed) vs predictions (red) on the 1024 PNG."""
    import cv2
    import matplotlib.patches as patches
    img = cv2.imread(f'{png_dir}/{image_id}.png', 0)
    ax = ax or plt.subplots(figsize=(8, 8))[1]
    ax.imshow(img, cmap='gray'); ax.axis('off')
    for r in gt[(gt.image_id == image_id) & (gt.class_id != NO_FINDING)].itertuples():
        x, y, w, h = r.x_min * IMG_SIZE, r.y_min * IMG_SIZE, (r.x_max - r.x_min) * IMG_SIZE, (r.y_max - r.y_min) * IMG_SIZE
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec='lime', lw=1.5, ls='--'))
        ax.text(x, y + h + 12, f'GT {CLASSES[r.class_id][:14]}', color='lime', fontsize=7)
    for r in preds[(preds.image_id == image_id) & (preds.class_id != NO_FINDING) & (preds.score >= score_thr)].itertuples():
        x, y, w, h = r.x_min * IMG_SIZE, r.y_min * IMG_SIZE, (r.x_max - r.x_min) * IMG_SIZE, (r.y_max - r.y_min) * IMG_SIZE
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, ec='red', lw=1.2))
        ax.text(x, y - 4, f'{CLASSES[r.class_id][:14]} {r.score:.2f}', color='red', fontsize=7)
    ax.set_title(title or image_id[:12], fontsize=9)
    return ax


def image_level_scores(res):
    """Per-image recall of GT findings (at any score) - used to pick best / worst examples."""
    g = res['gt'][res['gt'].class_id != NO_FINDING]
    return g.assign(hit=g.det_score.notna()).groupby('image_id').hit.mean().sort_values()


def show_examples(res, preds, png_dir, n=4, score_thr=0.2, seed=0):
    """Worst, best and random abnormal images side by side."""
    s = image_level_scores(res)
    picks = {'Worst recall': s.index[:n], 'Best recall': s.index[-n:],
             'Random': s.sample(min(n, len(s)), random_state=seed).index}
    for name, ids in picks.items():
        fig, axes = plt.subplots(1, len(ids), figsize=(5 * len(ids), 5))
        for ax, i in zip(np.ravel(axes), ids):
            show_predictions(i, res['gt'], preds, png_dir, score_thr, ax, f'{i[:10]} recall={s[i]:.2f}')
        fig.suptitle(f'{name} (GT green dashed, predictions red, score >= {score_thr})'); plt.tight_layout(); plt.show()


def full_detection_report(gt, preds, png_dir, name, score_thr=0.2, examples=True):
    """Everything for one detector / pipeline stage: mAP, PR curves, confusion, recall breakdown, examples."""
    print(f'===== {name} =====')
    res = evaluate(gt, preds)
    display_ = _display()
    display_(res['ap'].round(3).to_frame('AP'))
    plot_pr_curves(res, title=name)
    cm = detection_confusion(gt, preds, score_thr=score_thr)
    plot_confusion(cm, f'{name} - detection confusion (score >= {score_thr}, rows normalized)')
    by_size, by_rads, counts = recall_breakdown(res)
    print('Recall by size (any score):'); display_(by_size)
    print('Recall by n_rads (any score):'); display_(by_rads)
    if examples:
        show_examples(res, preds, png_dir, score_thr=score_thr)
    return res


def _display():
    try:
        from IPython.display import display
        return display
    except ImportError:
        return print


# =============================================================================
# Classifier reports
# =============================================================================
def _binary_counts(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv,
                F1=2 * ppv * sens / max(ppv + sens, 1e-9), accuracy=(tp + tn) / len(y))


def binary_report(y, p, sens_targets=(0.95, 0.99), name='Binary head'):
    """ROC/PR curves, AUC, and sensitivity/specificity/confusion at several thresholds:
    Youden's J (best balance) and the highest thresholds that reach each target sensitivity."""
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    auc, ap = roc_auc_score(y, p), average_precision_score(y, p)
    fpr, tpr, thr = roc_curve(y, p)
    thresholds = {'Youden J': thr[np.argmax(tpr - fpr)], '0.5': 0.5}
    for t in sens_targets:
        thresholds[f'sens>={t}'] = thr[np.argmax(tpr >= t)]
    table = pd.DataFrame({k: _binary_counts(y, (p >= v).astype(int)) | {'threshold': v}
                          for k, v in thresholds.items()}).T
    print(f'{name}: ROC AUC = {auc:.4f} | PR AUC (AP) = {ap:.4f} | positives = {y.mean():.1%}')

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], 'k--', lw=0.8)
    for k, v in thresholds.items():
        r = table.loc[k]; ax[0].scatter(1 - r.specificity, r.sensitivity, s=25, label=f'{k} ({v:.3f})')
    ax[0].set_xlabel('1 - specificity'); ax[0].set_ylabel('sensitivity'); ax[0].set_title(f'ROC (AUC={auc:.3f})'); ax[0].legend(fontsize=8)
    prec, rec, _ = precision_recall_curve(y, p)
    ax[1].plot(rec, prec); ax[1].set_xlabel('recall'); ax[1].set_ylabel('precision'); ax[1].set_title(f'PR (AP={ap:.3f})')
    ax[2].hist(p[y == 0], bins=50, alpha=0.6, label='normal', density=True)
    ax[2].hist(p[y == 1], bins=50, alpha=0.6, label='abnormal', density=True)
    ax[2].set_title('Score distribution by true label'); ax[2].legend(); ax[2].set_yscale('log')
    plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(1, len(thresholds), figsize=(4 * len(thresholds), 3.5))
    for axx, (k, v) in zip(axes, thresholds.items()):
        r = table.loc[k]
        m = np.array([[r.TN, r.FP], [r.FN, r.TP]], float)
        axx.imshow(m, cmap='Blues')
        for i in range(2):
            for j in range(2):
                axx.text(j, i, int(m[i, j]), ha='center', va='center', fontsize=12)
        axx.set_xticks([0, 1]); axx.set_xticklabels(['pred normal', 'pred abnormal'])
        axx.set_yticks([0, 1]); axx.set_yticklabels(['normal', 'abnormal'])
        axx.set_title(f'{k}: t={v:.3f}\nsens={r.sensitivity:.3f} spec={r.specificity:.3f}', fontsize=9)
    plt.tight_layout(); plt.show()
    return table.astype({'TP': int, 'FP': int, 'TN': int, 'FN': int}).round(4)


def multilabel_report(soft, probs):
    """Per-class quality of the multilabel head.

    soft: DataFrame (images x 14) of radiologist vote fractions {0, 1/3, 2/3, 1}
    probs: same shape, predicted p(class)
    - AUC / AP with 'any radiologist marked it' as positive
    - AUC with 'at least 2 radiologists' as positive
    - mean predicted p by vote level: should rise 0 -> 1/3 -> 2/3 -> 1 if the head learned agreement
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows, by_vote = {}, {}
    for c in range(14):
        y_any = (soft.iloc[:, c] > 0).astype(int); y_agr = (soft.iloc[:, c] >= 0.6).astype(int)
        p = probs.iloc[:, c]
        rows[CLASSES[c]] = dict(
            prevalence=y_any.mean(),
            AUC_any=roc_auc_score(y_any, p) if 0 < y_any.sum() < len(y_any) else np.nan,
            AP_any=average_precision_score(y_any, p) if y_any.sum() else np.nan,
            AUC_agreed=roc_auc_score(y_agr, p) if 0 < y_agr.sum() < len(y_agr) else np.nan)
        by_vote[CLASSES[c]] = p.groupby(soft.iloc[:, c].round(2)).mean()
    tab = pd.DataFrame(rows).T
    votes = pd.DataFrame(by_vote).T
    votes.columns = [f'mean p | votes={v:.2f}' for v in votes.columns]

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    tab[['AUC_any', 'AUC_agreed']].plot.barh(ax=ax[0]); ax[0].invert_yaxis(); ax[0].set_xlim(0.5, 1)
    ax[0].set_title('Multilabel head: AUC per class')
    ax[1].plot(range(votes.shape[1]), votes.T.values, marker='o')      # one line per class, x = vote level
    ax[1].set_xticks(range(votes.shape[1])); ax[1].set_xticklabels(['0', '1/3', '2/3', '1'][:votes.shape[1]])
    ax[1].legend(votes.index, fontsize=6, ncol=2)
    ax[1].set_xlabel('fraction of radiologists who marked the class'); ax[1].set_ylabel('mean predicted p')
    ax[1].set_title('Does p rise with radiologist agreement? (one line per class)')
    plt.tight_layout(); plt.show()
    return tab.round(3), votes.round(3)


# =============================================================================
# Fusion and soft combine
# =============================================================================
def top_k_per_image(df, k=100, min_score=0.001):
    """Keep the k highest-scoring predictions per image (speeds up fusion and tuning)."""
    df = df[df.score >= min_score]
    return df.sort_values('score', ascending=False).groupby('image_id').head(k).reset_index(drop=True)


def wbf_models(pred_dfs, weights=None, iou_thr=0.5, skip_box_thr=0.001, image_ids=None):
    """Weighted boxes fusion of several prediction DataFrames, image by image (per class inside WBF)."""
    from ensemble_boxes import weighted_boxes_fusion
    groups = [d.groupby('image_id') for d in pred_dfs]
    ids = image_ids if image_ids is not None else sorted(set().union(*[set(d.image_id) for d in pred_dfs]))
    out = []
    for img in ids:
        boxes, scores, labels = [], [], []
        for g in groups:
            if img in g.groups:
                p = g.get_group(img)
                boxes.append(p[BOX].values.clip(0, 1).tolist()); scores.append(p.score.tolist()); labels.append(p.class_id.tolist())
            else:
                boxes.append(np.zeros((0, 4)).tolist()); scores.append([]); labels.append([])
        if sum(len(s) for s in scores) == 0:
            continue
        b, s, l = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        out.append(pd.DataFrame({'image_id': img, 'class_id': l.astype(int), 'score': s,
                                 'x_min': b[:, 0], 'y_min': b[:, 1], 'x_max': b[:, 2], 'y_max': b[:, 3]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=PRED_COLS)


def wbf_per_class(pred_dfs, class_weights, iou_thr=0.5, skip_box_thr=0.001, image_ids=None):
    """WBF with its own model weights for every class.

    class_weights: 14 entries, one list of model weights per class (e.g. [[2, 1], [1, 2], ...]).
    WBF never merges boxes of different classes, so fusing each group of classes that share
    the same weights separately gives exactly the same result as one fusion per class.
    """
    groups = {}
    for c, w in enumerate(class_weights):
        groups.setdefault(tuple(float(x) for x in w), []).append(c)
    out = []
    for w, cs in groups.items():
        sub = [d[d.class_id.isin(cs)] for d in pred_dfs]
        out.append(wbf_models(sub, weights=list(w), iou_thr=iou_thr, skip_box_thr=skip_box_thr, image_ids=image_ids))
    return pd.concat(out, ignore_index=True)


# =============================================================================
# Per-class selection with an honest check
# =============================================================================
def match_table(gt, preds, iou_thr=0.4, classes=range(14)):
    """TP flag for every prediction.

    VOC matching happens inside each image, so the TP flags do not depend on which other
    images are evaluated. AP on any subset of images can then be computed from this table
    without matching again (see ap_from_table), which makes fold-by-fold checks cheap.
    """
    parts = []
    for c in classes:
        gt_c, pr_c = gt[gt.class_id == c], preds[preds.class_id == c]
        if len(pr_c) == 0:
            continue
        if len(gt_c) == 0:
            t = pr_c[['image_id', 'class_id', 'score']].sort_values('score', ascending=False).copy()
            t['tp'] = False
        else:
            pr_sorted, tp, _ = _match_class(gt_c, pr_c, iou_thr)
            t = pr_sorted[['image_id', 'class_id', 'score']].copy()
            t['tp'] = tp
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=['image_id', 'class_id', 'score', 'tp'])
    return pd.concat(parts, ignore_index=True)


def ap_from_table(tab, gt, image_ids=None, classes=range(14)):
    """AP per class (Series indexed by class name) from a match_table, optionally on a subset of images."""
    if image_ids is not None:
        tab = tab[tab.image_id.isin(image_ids)]
        gt = gt[gt.image_id.isin(image_ids)]
    n_gt = gt.class_id.value_counts()
    by_cls = dict(tuple(tab.groupby('class_id')))
    ap = {}
    for c in classes:
        n = int(n_gt.get(c, 0))
        if n == 0:
            ap[ALL_CLASSES[c]] = np.nan
            continue
        t = by_cls.get(c)
        if t is None or len(t) == 0:
            ap[ALL_CLASSES[c]] = 0.0
            continue
        tp = t.sort_values('score', ascending=False, kind='mergesort').tp.values.astype(bool)
        ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
        ap[ALL_CLASSES[c]] = voc_ap(ctp / n, ctp / np.maximum(ctp + cfp, 1e-9))
    return pd.Series(ap)


def select_per_class(cands, gt, fold_of, classes=range(14), iou_thr=0.4):
    """Pick, for every class, the candidate prediction set with the highest AP.

    cands:   {name: predictions}; the candidates differ in one setting (e.g. model weights or beta)
    fold_of: Series image_id -> fold, for the evaluated images

    Returns a dict with:
      ap     - DataFrame class x candidate, AP on all images
      choice - Series class name -> chosen candidate (made on all images)
      check  - one row per fold: choices made on the OTHER folds, measured on this fold, for
               (a) the best single candidate for all classes and (b) a separate choice per class.
               Per-class choices are only worth keeping if (b) beats (a) here.
    """
    tabs = {k: match_table(gt, p, iou_thr, classes) for k, p in cands.items()}
    ap = pd.DataFrame({k: ap_from_table(t, gt, None, classes) for k, t in tabs.items()})
    choice = ap.fillna(-1).idxmax(axis=1)          # a class with no GT here keeps the first candidate
    rows = []
    folds_here = sorted(pd.unique(fold_of))
    for k in folds_here if len(folds_here) > 1 else []:
        tr, te = fold_of.index[fold_of != k], fold_of.index[fold_of == k]
        ap_tr = pd.DataFrame({n: ap_from_table(t, gt, tr, classes) for n, t in tabs.items()})
        ap_te = pd.DataFrame({n: ap_from_table(t, gt, te, classes) for n, t in tabs.items()})
        ch = ap_tr.fillna(-1).idxmax(axis=1)
        best_single = ap_tr.mean().idxmax()
        per_class = pd.Series({c: ap_te.loc[c, ch[c]] for c in ap_te.index})
        rows.append(dict(fold=k, single_choice=best_single, map_single=ap_te[best_single].mean(),
                         map_per_class=per_class.mean()))
    check = pd.DataFrame(rows)
    if len(check):
        check['gain'] = check.map_per_class - check.map_single
    return dict(ap=ap, choice=choice, check=check)


def assemble_per_class(cands, choice):
    """Predictions of class c taken from the candidate chosen for c; No finding rows from the first candidate."""
    parts = []
    for c, name in enumerate(CLASSES):
        p = cands[choice[name]]
        parts.append(p[p.class_id == c])
    first = next(iter(cands.values()))
    parts.append(first[first.class_id == NO_FINDING])
    return pd.concat(parts, ignore_index=True)


def soft_combine(boxes, cls, alpha=1.0, beta=None, t_low=0.0):
    """Combine fused boxes with the classifier heads.

    boxes: fused predictions (findings only)
    cls:   DataFrame indexed by image_id with p_abn and p_c0..p_c13
    For every image: add No finding with score 1 - p_abn; drop all boxes if p_abn < t_low;
    otherwise score' = score * p_abn**alpha * p_class**beta[class].
    """
    beta = np.zeros(14) if beta is None else np.asarray(beta, float)
    b = boxes.merge(cls[['p_abn']], left_on='image_id', right_index=True, how='left')
    pc = cls[[f'p_c{c}' for c in range(14)]].values
    row = cls.index.get_indexer(b.image_id)
    p_class = pc[row, b.class_id.values]
    b['score'] = b.score * b.p_abn ** alpha * p_class ** beta[b.class_id.values]
    b = b[b.p_abn >= t_low].drop(columns='p_abn')
    nf = pd.DataFrame({'image_id': cls.index, 'class_id': NO_FINDING, 'score': 1 - cls.p_abn.values,
                       'x_min': 0.0, 'y_min': 0.0, 'x_max': 1.0, 'y_max': 1.0})
    return pd.concat([b[PRED_COLS], nf], ignore_index=True)


def add_partner_copies(preds, factor=0.5, pairs=PARTNER_PAIRS, nms_iou=0.5):
    """For each box of a double-labeled pair, add a copy as the partner class with score * factor,
    then run per-class NMS so a copy never duplicates an existing prediction."""
    copies = []
    for a, b in pairs:
        for src, dst in ((a, b), (b, a)):
            c = preds[preds.class_id == src].copy()
            c['class_id'] = dst; c['score'] *= factor
            copies.append(c)
    out = pd.concat([preds] + copies, ignore_index=True)
    return nms_per_class(out, nms_iou)


def double_label_coverage(gt, preds, iou_thr=0.4, score_thr=0.1, pairs=PARTNER_PAIRS):
    """For GT regions labeled with both classes of a pair: how often does the model predict both?"""
    rows = []
    for a, b in pairs:
        ga, gb = gt[gt.class_id == a], gt[gt.class_id == b]
        both = one = none = 0
        for img, g1 in ga.groupby('image_id'):
            g2 = gb[gb.image_id == img]
            if len(g2) == 0:
                continue
            ious = iou_matrix(g1[BOX].values, g2[BOX].values)
            p = preds[(preds.image_id == img) & (preds.score >= score_thr)]
            for i, j in zip(*np.where(ious >= iou_thr)):       # a double-labeled region
                region = g1[BOX].values[i]
                hit = {k: (iou_matrix(region, p.loc[p.class_id == k, BOX].values) >= iou_thr).any()
                       for k in (a, b)}
                s = sum(hit.values()); both += s == 2; one += s == 1; none += s == 0
        tot = max(both + one + none, 1)
        rows.append(dict(pair=f'{CLASSES[a]} + {CLASSES[b]}', regions=both + one + none,
                         predicted_both=both / tot, predicted_one=one / tot, predicted_none=none / tot))
    return pd.DataFrame(rows).round(3)


# =============================================================================
# Submission
# =============================================================================
def prediction_strings(preds, meta):
    """Build the PredictionString per image in original pixel coordinates."""
    size = meta.set_index('image_id')[['rows', 'cols']]
    out = {}
    for img, g in preds.sort_values('score', ascending=False).groupby('image_id'):
        rows, cols = size.loc[img]
        parts = []
        for r in g.itertuples():
            if r.class_id == NO_FINDING:
                parts.append(f'14 {r.score:.5f} 0 0 1 1')
            else:
                parts.append(f'{r.class_id} {r.score:.5f} {r.x_min * cols:.0f} {r.y_min * rows:.0f} '
                             f'{r.x_max * cols:.0f} {r.y_max * rows:.0f}')
        out[img] = ' '.join(parts)
    return out


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)
