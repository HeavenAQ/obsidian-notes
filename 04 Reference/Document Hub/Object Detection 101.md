---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-06-10T14:41:00
Status: Not started
Last updated time: 2026-06-14T12:25:00
Last edited by: Heaven Chen
Category:
  - CV
  - Object Detection
---
# ROI Align

- Region of Interest Align
- A feature extraction operation used in object detection and segmentaion

### 1. Why is ROI align needed?

```javascript
Image Input -> CNN Backbone -> Feature Map -> ROIs
```

- So after the CNN backbone generates the feature map, the CNN model will generate regions of interest (ROIs) that may look like:
```javascript
ROI1: 30 x 30 Person 
ROI2: 10 x 10 Dog
...
```
- The problem is that, for the detection head to work, fixed-size inputs are needed. Therefore, ROI aligns create fixed-size feature map for each ROI
```javascript
ROIS -> ROI Align -> fixed-size feature map
----------------------------------------------
ROI region
+-----------+
|           |
|           |
+-----------+

ROI Align output
+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+
```

### 2. How does ROI Align work?

- ROI Align uses bilinear interpolation to interpolate nearby feature values given the floating-point coordinates
- This way, the spatial features are kept.
- For example, we have:

```javascript
(4, 6) Q11 = 10 ----------------- Q21 = 20 (5, 6)
        |                          |
        |       P (4.3, 6.7)       |
        |                          |
(4, 7) Q12 = 30 ----------------- Q22 = 40 (5, 7)
```

- `sampled value = weighted average of the nearby values`
- **Weights:**
    - `x = 5 - 4 = 1`, `y = 7 - 6 = 1` 
    - Q11: (1 - 0.3) * (1 - 0.7) = 0.21
    - Q12: (1 - 0.3) * (1 - 0.3) = 0.49
    - Q21: (1 - 0.7) * (1 - 0.7) = 0.09
    - Q22: (1 - 0.7) * (1 - 0.3) = 0.21
- **Value:**
    - 10 * 0.21 + 20 * 0.09 + 30 * 0.49 + 40 * 0.21 = 27

# Analysis

Research findings, data insights, and key considerations

# Recommendations

Proposed solutions, strategies, and next steps

# Implementation

Action items, timeline, and resource requirements