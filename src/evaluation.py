"""Evaluation metrics for structured aerial scene analysis.

Computes 7 metrics comparing model predictions against ground truth:
schema compliance, scene type accuracy, ROUGE-1/2/L, BERTScore, Object Mention F1.
"""

from collections import Counter, defaultdict

import numpy as np

from .inference import validate_schema


def extract_object_types(objects_list):
    """Extract set of object type labels from an objects list."""
    if not isinstance(objects_list, list):
        return set()
    return {obj["type"] for obj in objects_list if isinstance(obj, dict) and "type" in obj}


def compute_all_metrics(predictions, device="cuda"):
    """Compute all evaluation metrics on a list of prediction dicts.

    Each prediction dict must have keys:
        image_file, scene_type_gt, caption_gt, objects_gt,
        infrastructure_gt, terrain_gt, raw_prediction

    Returns a dict with metric values and a _details sub-dict for inspection.
    """
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_fn

    # ── Parse predictions ──
    parsed_preds = []
    schema_results = []
    for pred in predictions:
        obj, error = validate_schema(pred["raw_prediction"])
        parsed_preds.append(obj)
        schema_results.append({
            "image": pred["image_file"],
            "valid": error is None,
            "error": error,
        })

    n_valid = sum(1 for r in schema_results if r["valid"])
    compliance_rate = n_valid / len(schema_results) * 100

    # ── Scene Type Accuracy ──
    correct = 0
    confusion = defaultdict(Counter)
    for pred, parsed in zip(predictions, parsed_preds):
        gt = pred["scene_type_gt"]
        predicted = parsed.get("scene_type", "<missing>") if parsed else "<parse_fail>"
        confusion[gt][predicted] += 1
        if predicted == gt:
            correct += 1
    scene_accuracy = correct / len(predictions) * 100

    # ── ROUGE ──
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, parsed in zip(predictions, parsed_preds):
        gt_caption = pred["caption_gt"]
        pred_caption = parsed.get("caption", "") if parsed else ""
        scores = scorer.score(gt_caption, pred_caption)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    # ── BERTScore ──
    gt_captions = [p["caption_gt"] for p in predictions]
    pred_captions = [
        parsed.get("caption", "") if parsed else ""
        for parsed in parsed_preds
    ]
    P, R, F1 = bert_score_fn(
        pred_captions, gt_captions, lang="en", verbose=False, device=device
    )

    # ── Object Mention F1 ──
    per_sample_f1 = []
    per_type_tp = Counter()
    per_type_fp = Counter()
    per_type_fn = Counter()
    for pred, parsed in zip(predictions, parsed_preds):
        gt_types = extract_object_types(pred["objects_gt"])
        pred_types = extract_object_types(parsed.get("objects", [])) if parsed else set()
        tp = gt_types & pred_types
        fp = pred_types - gt_types
        fn = gt_types - pred_types
        precision = len(tp) / len(pred_types) if pred_types else 0.0
        recall = len(tp) / len(gt_types) if gt_types else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_sample_f1.append(f1)
        for t in tp:
            per_type_tp[t] += 1
        for t in fp:
            per_type_fp[t] += 1
        for t in fn:
            per_type_fn[t] += 1

    return {
        "schema_compliance_pct": round(compliance_rate, 1),
        "scene_type_accuracy_pct": round(scene_accuracy, 1),
        "rouge1_f1": round(float(np.mean(rouge_scores["rouge1"])), 4),
        "rouge2_f1": round(float(np.mean(rouge_scores["rouge2"])), 4),
        "rougeL_f1": round(float(np.mean(rouge_scores["rougeL"])), 4),
        "bertscore_f1": round(float(F1.mean()), 4),
        "object_mention_f1": round(float(np.mean(per_sample_f1)), 4),
        "_details": {
            "schema_results": schema_results,
            "confusion": {gt: dict(preds) for gt, preds in confusion.items()},
            "rouge_per_sample": rouge_scores["rougeL"],
            "bertscore_per_sample": F1.tolist(),
            "object_f1_per_sample": per_sample_f1,
            "parsed_preds": parsed_preds,
        },
    }
