---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-06T19:03:00
Status: Done
Last updated time: 2026-07-06T19:03:00
Last edited by: Heaven Chen
Category:
  - Object-Detection
  - Pose-Estimation
---
Date: 2026-06-21 · Remote host: `sb-00477-login` · Remote repo: `~/heaven/rf-detr-preview` · Dataset: `~/heaven/pose-person-hand-detection-model/datasets/coco-full/train/_annotations.coco.json`

## Files changed

I did not overwrite the original training script. New remote script:

```bash
~/heaven/rf-detr-preview/scripts/train_coco17_person_hand_keypoints_with_labels.py
```

Original left unchanged: `~/heaven/rf-detr-preview/scripts/train_coco17_person_hand_keypoints.py`

Run the new script with the same arguments as the original, for example:

```bash
cd ~/heaven/rf-detr-preview
uv run python scripts/train_coco17_person_hand_keypoints_with_labels.py \
  --dataset-dir /path/to/dataset \
  --output-dir /path/to/output \
  --wandb-entity your-wandb-entity \
  --project rf-detr-coco17-person-hand
```

## Visualization change

The qualitative W&B visualization now draws class labels on bbox overlays.

- Ground-truth boxes are labeled with the class name (e.g. `person` or `hand`).
- Prediction boxes are labeled with the class name plus confidence when available (e.g. `person 0.87`).
- Prediction boxes are filtered with class-wise NMS before drawing/counting in `--visualization-only`.
- The shared OpenCV drawing helper still draws the same bbox, skeleton, and keypoint dots as before.

Implementation details in the new script:

- `_draw_one_instance(...)` now accepts `label: str | None`.
- The helper draws a filled label background at the top-left of the bbox using the class color.
- `draw_gt(...)` passes `label=meta["name"]`.
- `draw_pred(...)` reads `prediction.detection_confidence`, falling back to `prediction.confidence`, and passes `label=f"{class} {score:.2f}"` when a score is present.
- `draw_pred(...)` runs `torchvision.ops.nms` separately per predicted class before drawing.
- New CLI flag: `--visualization-nms-iou`, default `0.5`. Set it negative to disable visualization NMS.

Example with stronger duplicate suppression:

```bash
cd ~/heaven/rf-detr-preview
uv run python scripts/train_coco17_person_hand_keypoints_with_labels.py \
  --dataset-dir /path/to/dataset \
  --output-dir /path/to/output \
  --visualization-only \
  --pretrain-weights /path/to/checkpoint.pth \
  --visualization-nms-iou 0.4
```

Validation performed:

```bash
cd ~/heaven/rf-detr-preview
.venv/bin/python -m py_compile scripts/train_coco17_person_hand_keypoints_with_labels.py
.venv/bin/python - <<'PY'
from pathlib import Path
import ast
p = Path("scripts/train_coco17_person_hand_keypoints_with_labels.py")
ast.parse(p.read_text())
print("ast ok", p)
PY
```

Both checks passed.

## Dataset findings relevant to person-only duplicate boxes

The COCO training annotation file contains:

- Images: 118,287
- Annotations: 418,937
- `person`: 262,465 annotations
- `hand`: 156,472 annotations

Category schema:

- `person`: category id 1, 17 keypoints, 19 skeleton links
- `hand`: category id 2, 0 keypoints, 0 skeleton links

RF-DETR schema inferred in the repo venv:

```plain text
class_names ['person', 'hand']
num_keypoints_per_class [17, 0]
```

Visible keypoint statistics:

- `person`: median 5 visible keypoints, mean 6.26, max 17, zero-visible annotations 112,652
- `hand`: all 156,472 hand annotations have zero visible keypoints because the category has no keypoint schema

Near-duplicate annotations in the train split:

- Same-image same-class person bbox pairs with IoU >= 0.80: 52
- Same-image same-class person bbox pairs with IoU >= 0.98: 3
- No same-image same-class hand bbox pairs were found above IoU >= 0.80 in this check

This means upstream annotation duplication exists for `person`, but it is sparse relative to the dataset size. It can explain occasional duplicate person predictions on specific images, but it is unlikely to be the only cause if duplicates are frequent.

## RF-DETR keypoint docs checked

I found one keypoint reference doc: `~/heaven/rf-detr-preview/docs/rfdetr_keypoint_detection.md`. Relevant points:

- RF-DETR is DETR-style set prediction with Hungarian one-to-one matching during training.
- Postprocess applies sigmoid scores and global top-k over the flattened query-by-class grid. It does not apply NMS.
- Group-DETR uses one-to-many groups during training for stability, while inference uses the first group.
- Keypoint predictions are class-padded by `num_keypoints_per_class`.
- Slot 7 of active keypoints is summed as a keypoint-to-class logit boost.
- Keypoint losses and matching costs apply only to classes with active keypoints.
- `RFDETRKeypointPreview` defaults to a person keypoint model, but this training script overrides the schema from the dataset metadata.
- The GPU/Kornia backend does not transform keypoints in this preview; keypoint training should use the CPU/Albumentations backend.

## Most likely causes for duplicate person boxes but not hands

1. **Person has active keypoint supervision; hand is box-only in this dataset.** The inferred schema is `[17, 0]`, so person detections receive keypoint losses, keypoint matching costs, and keypoint-to-class logit boost. Hands do not. The score path for person therefore has an extra keypoint-driven confidence term that can keep multiple person queries above threshold for the same person.
2. **RF-DETR postprocess does not suppress duplicates with NMS.** Postprocess takes global top-k detections from query/class scores. If two queries survive with high `person` scores on the same person, both can be displayed. Hands may not show this because they lack keypoint score boost and have much smaller boxes.
3. **Some person annotations are duplicated or near-duplicated in the training set.** 52 person bbox pairs at IoU >= 0.80 and 3 at IoU >= 0.98. DETR-style matching can learn to allocate separate queries when duplicate GT exists. Not observed for hands.
4. **Many person annotations have weak or missing keypoints.** 112,652 person annotations have zero visible keypoints. These still provide box/class supervision while visible-keypoint losses are masked — a mixed signal that makes the person score path less uniform than the hand path.
5. **Threshold/top-k settings can expose duplicates.** Visualization default is `--threshold 0.4`. With no NMS, lowering the threshold makes duplicate query predictions more visible; raising it may hide but not fix them.

## Follow-up checks recommended before retraining

- Visualize several duplicate-prediction images with the new class labels and compare against GT counts in the W&B table.
- Sanitize or review the 52 near-duplicate person annotation pairs if duplicate predictions correlate with those images.
- Decide whether hands should be a keypoint class. If yes, the dataset category needs hand keypoint names and populated hand keypoints; currently hands are trained as bbox-only objects.
- If duplicates are only a visualization/inference issue, add an optional class-wise NMS or weighted boxes fusion step to the visualization/inference path only. Do not put that into training unless the RF-DETR pipeline explicitly supports it.
- Keep using the Albumentations/CPU augmentation path for keypoint training, because the docs state the GPU/Kornia backend does not transform keypoints.