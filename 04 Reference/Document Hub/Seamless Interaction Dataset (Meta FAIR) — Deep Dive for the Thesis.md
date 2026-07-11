---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-11T15:47:00
Status: Done
Last updated time: 2026-07-11T15:47:00
Last edited by: Heaven Chen
Category:
  - CV
  - Pose Estimation
  - Survey
---
> **Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset.** Agrawal et al., **Meta FAIR / Reality Labs**, 2025. License **CC-BY-NC 4.0**. [Paper](https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset/) · [Repo](https://github.com/facebookresearch/seamless_interaction) · [HuggingFace](https://huggingface.co/datasets/facebook/seamless-interaction) · [Blog](https://ai.meta.com/blog/seamless-interaction-natural-conversational-dynamics/).

> **This is the primary dataset for the thesis.**

## 1. Overview

**4,000+ hours** of face-to-face **dyadic** interaction from **4,000+ participants** across diverse contexts — built to train AI that understands human conversational dynamics (virtual agents, telepresence, multimodal analysis). Two labels:

- **Improvised** — guided prompts / predefined scenarios with at least one professional actor.
- **Naturalistic** — prompted but spontaneous conversations between ordinary people.

Standard **train / dev / test** splits. Whole dataset ≈ **27 TB**; organized into self-contained **~50 GB batches** (dev/test = 5 batches each ≈ 500 GB; train = 200+ batches ≈ 20 TB+). WebDataset tarballs on HF; fine-grained files on S3.

## 2. Why it fits this thesis (the key point)

It ships — **precomputed** — almost every input my pipeline needs, so I don't have to run pose/mesh estimation myself:

| Thesis need | Provided in the dataset |
| --- | --- |
| Self-contact / self-adaptor detection | **SMPL-H** body + **left/right hand** pose (30 Hz) → compute Euclidean-near + geodesic-far contact directly |
| Detection / localisation layer | Face & body **keypoints**  • bounding boxes (30 Hz) |
| Discourse-planning / disfluency signal | **Time-aligned transcripts** (JSONL) + **VAD** (100 Hz) → filled/silent pauses, repairs |
| Behavioral context | **Imitator movement features**: FAU (action units), gaze, head, emotion arousal/valence |
| Labels & validation | Human **internal-state** annotations (1P / 3P), rationale, visual annotations |
| Participant covariates | Personality (**BFI-2**), relationships, Interpersonal-Circumplex (IPC) classifications |

**Implication:** the "detect → 3D body+hands → self-contact → align to speech" pipeline can run largely on *provided* features. My contribution shifts to the **self-adaptor definition/segmentation** and the **behavioral link** to planning difficulty — not re-implementing pose estimation.

## 3. Modalities & features

| Modality | Format | Rate |
| --- | --- | --- |
| Video (HD face-to-face) | MP4 / H.264 | 30 fps, 1080p |
| Audio (denoised, separate channels) | WAV | 48 kHz, 16-bit |
| Transcript (time-aligned) | JSONL | — |
| SMPL-H body model params | NPZ | 30 Hz |
| Keypoints (face + body) + boxes | NPZ | 30 Hz |
| VAD (voice activity) | JSONL | 100 Hz |
| Imitator movement features | NPZ | 30 Hz |
| Human annotations | JSON | — |

**NPZ keys** (per interaction file):

```plain text
smplh:body_pose, smplh:global_orient, smplh:left_hand_pose, smplh:right_hand_pose,
smplh:translation, smplh:is_valid
boxes_and_keypoints:box, boxes_and_keypoints:keypoints, boxes_and_keypoints:is_valid_box
movement:FAUToken, movement:FAUValue, movement:emotion_arousal, movement:emotion_valence,
movement:emotion_scores, movement:expression, movement:gaze_encodings, movement:head_encodings,
movement:frame_latent, movement:hypernet_features, movement:is_valid
```

JSON keys: `id`, `metadata:transcript`, `metadata:vad`. Emotion scores span 8 categories (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise); FAUValue covers 24 action units (BrowLowerer, LipCornerPuller, JawDrop, …).

## 4. Structure, splits & naming

```plain text
improvised | naturalistic
  └ dev | test | train
      ├ 1P-IS/ 1P-R/ 3P-IS/ 3P-R/ 3P-V/   (first/third-party internal-state + rationale + visual annotations)
      ├ audio/  video/  transcript/  vad/
      ├ boxes_and_keypoints/{box,is_valid_box,keypoints}
      ├ smplh/{body-pose,global_orient,left_hand_pose,right_hand_pose,translation,is_valid}
      └ movement/{emotion_*,expression,FAUToken,FAUValue,gaze_encodings,head_encodings,…}
```

File naming: `V<vendor>_S<session>_I<interaction>_P<participant>` (e.g. `V00_S0809_I00000582_P0947`). Metadata CSVs: `filelist.csv` (availability flags `has_annotation_1p/3p`, `has_imitator_movement`), `interactions.csv` (prompt hashes, IPC `ipc_a`/`ipc_b`), `participants.csv`, `relationships.csv`.

## 5. Access & code

```bash
git clone https://github.com/facebookresearch/seamless_interaction
cd seamless_interaction && pip install -e .
# interactive browser:
streamlit run src/seamless_interaction/app/Welcome.py
```

```python
# Fine-grained (S3): one interaction / pair / file id
from seamless_interaction.fs import SeamlessInteractionFS, DatasetConfig
from pathlib import Path

cfg = DatasetConfig(label="naturalistic", split="dev", preferred_vendors_only=True,
                    local_dir=Path.home()/"datasets/seamless_interaction")
fs = SeamlessInteractionFS(config=cfg)
fid = fs.sample_random_file_ids(num_samples=1)[0]
fs.gather_file_id_data_from_s3(fid)           # -> .wav/.mp4/.npz/.json

# Batch (HF WebDataset, ~1GB sample set)
fs.download_batch_from_hf(batch_idx=0, archive_list=[0])
```

```python
# Stream via HF datasets + WebDataset
from datasets import load_dataset
url = "https://huggingface.co/datasets/facebook/seamless-interaction/resolve/main/naturalistic/dev/0000/0000.tar"
ds = load_dataset("webdataset", data_files={"dev":[url]}, split="dev", streaming=True)
item = next(iter(ds))
rh = item["npz"]["smplh:right_hand_pose"]     # right-hand pose per frame -> feed self-contact
```

Self-adaptor pipeline note: reconstruct SMPL-H vertices from the provided `smplh:*` params, then apply the TUCH **Euclidean-near + geodesic-far** contact test (see the “Human Mesh Math” doc) to flag hand↔face/body touch per frame; segment temporal events; align to `metadata:transcript` + `metadata:vad`.

## 6. Mapping to the research questions

- **RQ1 (behavioural):** self-adaptor events (from SMPL-H self-contact) vs disfluency (transcripts + VAD) — co-occurrence & time-locked analysis, with internal-state annotations for validation.
- **RQ2 (methods / generalisation):** train/eval self-adaptor detection across **improvised vs naturalistic** and across **vendors/sites** — a built-in domain-shift proxy for the in-cabin target.
- **RQ3 (multimodal):** fuse visual self-adaptors with speech disfluency and the provided **movement/emotion** features to predict planning difficulty.

## 7. Caveats

- **License CC-BY-NC 4.0 (non-commercial)** — fine for the thesis, but the **in-cabin / DENSO commercial** application would need separate permission.
- **Metadata noise:** ~**10%** of interactions have timestamp errors (start/end, prompt–speech misalignment, off-by-one ordering); rare participant-ID duplication; only **active-time** segments released (no “meta time” yet).
- **Site variation:** recording quality, speaker bleed, acting quality differ across vendors.
- **Some movement features are proprietary** (“HyperModel”: `frame_latent`, `expression`, `hypernet_features`, alignment params) and not fully released; annotations cover only a **subset** of files (check `filelist.csv` flags).
- **Scale:** 27 TB total — work from dev/test batches; don't pull the whole set.

## 8. Sources

- [Paper — Seamless Interaction (Meta FAIR, 2025)](https://ai.meta.com/research/publications/seamless-interaction-dyadic-audiovisual-motion-modeling-and-large-scale-dataset/)
- [GitHub — facebookresearch/seamless_interaction](https://github.com/facebookresearch/seamless_interaction)
- [HuggingFace — facebook/seamless-interaction](https://huggingface.co/datasets/facebook/seamless-interaction)
- [Blog](https://ai.meta.com/blog/seamless-interaction-natural-conversational-dynamics/) · [Demo](https://www.aidemos.meta.com/seamless_interaction_dataset)