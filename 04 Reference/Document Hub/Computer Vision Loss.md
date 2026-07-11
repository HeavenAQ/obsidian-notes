---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-06-14T12:20:00
Status: Not-Started
Last updated time: 2026-06-14T23:37:00
Last edited by: Heaven Chen
Category:
  - CV
---
# Common Loss Functions in Computer Vision

## 1. Overview

In computer vision, the loss function should match the model output representation.

Common output types include:

```plain text
class probabilities
bounding boxes
segmentation masks
keypoint coordinates
keypoint heatmaps
image embeddings
generated / restored images
```

A model is trained by minimizing a loss function:

$$
\theta^{*}
=
\arg\min_{\theta}
\mathcal{L}
\left(
	f_{\theta}(x), y
\right)
$$

where:

- $x$ is the input image
- $y$ is the ground-truth label
- $f_{\theta}(x)$ is the model prediction
- $\theta$ represents the model parameters
- $\mathcal{L}$ is the loss function

The key idea is:

```plain text
classification output → classification loss
box output → box regression / IoU loss
mask output → pixel-wise / overlap loss
heatmap output → heatmap loss
coordinate output → coordinate regression loss
embedding output → metric learning loss
image output → pixel / perceptual / adversarial loss
```

---

## 2. Classification Losses

Classification losses are used when the model predicts a class label.

Common tasks include:

```plain text
image classification
object classification
pixel classification
multi-label classification
foreground / background classification
```

### 2.1 Cross-Entropy Loss

Cross-entropy is the most common loss for single-label classification.

For one sample with ground-truth class $c$, the loss is:

$$
\mathcal{L}_{CE}
=
-
\log
\hat{p}_{c}
$$

where:

- $\hat{p}_{c}$ is the predicted probability of the correct class.

For a one-hot target vector $y$:

$$
\mathcal{L}_{CE}
=
-
\sum_{c=1}^{C}
	y_c
	\log
	\hat{p}_c
$$

where:

- $C$ is the number of classes.
- $y_c$ is the ground-truth label for class $c$.
- $\hat{p}_c$ is the predicted probability for class $c$.

The predicted probability usually comes from softmax:

$$
\hat{p}_c
=
\frac{
	\exp(z_c)
}{
	\sum_{j=1}^{C}
	\exp(z_j)
}
$$

where:

- $z_c$ is the logit for class $c$.

Cross-entropy is used when each sample belongs to exactly one class.

Examples:

```plain text
ResNet image classification
ViT image classification
DETR object-query classification
semantic segmentation pixel classification
```

### 2.2 Binary Cross-Entropy Loss

Binary cross-entropy is used for binary classification.

$$
\mathcal{L}_{BCE}
=
-
\left[
	y
	\log
	\hat{p}
	+
	(1-y)
	\log
	\left(
		1-\hat{p}
	\right)
\right]
$$

where:

- $y \in \{0,1\}$ is the binary label.
- $\hat{p}$ is the predicted probability of the positive class.

For multi-label classification with $C$ independent labels:

$$
\mathcal{L}_{BCE}
=
-
\sum_{c=1}^{C}
\left[
	y_c
	\log
	\hat{p}_c
	+
	(1-y_c)
	\log
	\left(
		1-\hat{p}_c
	\right)
\right]
$$

Binary cross-entropy is commonly used for:

```plain text
multi-label classification
foreground / background prediction
mask prediction
binary segmentation
dense binary heatmaps
```

Difference from softmax cross-entropy:

```plain text
cross-entropy with softmax:
  classes are mutually exclusive

binary cross-entropy with sigmoid:
  classes are independent
```

### 2.3 Focal Loss

Focal loss is designed for class imbalance. It reduces the contribution of easy examples and focuses more on hard examples.

For binary classification:

$$
\mathcal{L}_{focal}
=
-
\alpha_t
\left(
	1-p_t
\right)^{\gamma}
\log
p_t
$$

where:

$$
p_t
=
\begin{cases}
	\hat{p}, & \text{if } y=1 \\
	1-\hat{p}, & \text{if } y=0
\end{cases}
$$

and:

- $\alpha_t$ balances positive and negative samples.
- $\gamma$ controls how strongly easy examples are down-weighted.
- $p_t$ is the predicted probability assigned to the correct class.

When $gamma=0$, focal loss becomes regular cross-entropy:

$$
\mathcal{L}_{focal}
=
-
\alpha_t
\log
p_t
$$

Focal loss is commonly used when there are many easy background examples.

Examples:

```plain text
RetinaNet
dense object detection
foreground / background classification
imbalanced segmentation
```

### 2.4 Label Smoothing Cross-Entropy

Label smoothing prevents the model from becoming too confident.

Instead of using a hard one-hot label, the target is softened:

$$
y_c^{smooth}
=
\begin{cases}
	1-\epsilon, & \text{if } c = c^{*} \\
	\frac{\epsilon}{C-1}, & \text{otherwise}
\end{cases}
$$

where:

- $c^{*}$ is the ground-truth class.
- $C$ is the number of classes.
- $\epsilon$ is the smoothing factor.

The loss is:

$$
\mathcal{L}_{LSCE}
=
-
\sum_{c=1}^{C}
	y_c^{smooth}
	\log
	\hat{p}_c
$$

Label smoothing is often used in image classification and transformer-based models.

---

## 3. Bounding Box Regression Losses

Bounding box losses are used in object detection.

A bounding box can be represented as:

$$
b
=
(x_1, y_1, x_2, y_2)
$$

or:

$$
b
=
(c_x, c_y, w, h)
$$

where:

- $(x_1, y_1)$ is the top-left corner.
- $(x_2, y_2)$ is the bottom-right corner.
- $(c_x, c_y)$ is the box center.
- $w$ and $h$ are width and height.

### 3.1 L1 Loss

L1 loss measures absolute coordinate difference.

$$
\mathcal{L}_{L1}
=
\left\lVert
	\hat{b}
	-
	b
\right\rVert_1
$$

Expanded:

$$
\mathcal{L}_{L1}
=
\sum_{i=1}^{4}
\left|
	\hat{b}_i
	-
	b_i
\right|
$$

where:

- $\hat{b}$ is the predicted box.
- $b$ is the ground-truth box.

L1 loss is robust and simple, but it does not directly measure box overlap.

It is commonly used in DETR-style detection together with GIoU loss.

### 3.2 L2 Loss / Mean Squared Error

L2 loss penalizes squared coordinate differences.

$$
\mathcal{L}_{L2}
=
\left\lVert
	\hat{b}
	-
	b
\right\rVert_2^2
$$

Expanded:

$$
\mathcal{L}_{L2}
=
\sum_{i=1}^{4}
\left(
	\hat{b}_i
	-
	b_i
\right)^2
$$

L2 loss penalizes large errors more strongly than L1 loss.

However, it can be sensitive to outliers and is less common than L1 or Smooth L1 for box regression.

### 3.3 Smooth L1 Loss

Smooth L1 loss behaves like L2 near zero and like L1 for large errors.

For error $x$:

$$
\operatorname{smooth}_{L1}(x)
=
\begin{cases}
	\frac{1}{2}
	\frac{x^2}{\beta},
	& \text{if } |x| < \beta \\
	|x|
	-
	\frac{1}{2}\beta,
	& \text{otherwise}
\end{cases}
$$

The full box loss is:

$$
\mathcal{L}_{smoothL1}
=
\sum_{i=1}^{4}
\operatorname{smooth}_{L1}
\left(
	\hat{b}_i - b_i
\right)
$$

where:

- $\beta$ controls the transition point between L2-like and L1-like behavior.

Smooth L1 is commonly used in two-stage detectors.

Examples:

```plain text
Fast R-CNN
Faster R-CNN
Mask R-CNN box regression
```

---

## 4. IoU-Based Box Losses

Coordinate losses such as L1 and L2 do not directly measure overlap. IoU-based losses optimize box overlap directly.

### 4.1 IoU Loss

Intersection over Union is:

$$
\operatorname{IoU}(A, B)
=
\frac{
	|A \cap B|
}{
	|A \cup B|
}
$$

where:

- $A$ is the predicted box.
- $B$ is the ground-truth box.
- $|A \cap B|$ is the intersection area.
- $|A \cup B|$ is the union area.

The IoU loss is:

$$
\mathcal{L}_{IoU}
=
1
-
\operatorname{IoU}(A, B)
$$

IoU loss directly rewards overlap.

However, when two boxes do not overlap, $operatorname{IoU}(A,B)=0$, so the gradient may be weak or uninformative.

### 4.2 Generalized IoU Loss

Generalized IoU improves IoU when boxes do not overlap.

$$
\operatorname{GIoU}(A, B)
=
\operatorname{IoU}(A, B)
-
\frac{
	|C \setminus (A \cup B)|
}{
	|C|
}
$$

where:

- $C$ is the smallest enclosing box containing both $A$ and $B$.
- $C \setminus (A \cup B)$ is the area inside $C$ but outside the union of $A$ and $B$.

The GIoU loss is:

$$
\mathcal{L}_{GIoU}
=
1
-
\operatorname{GIoU}(A, B)
$$

GIoU is commonly used in DETR-style object detectors.

Examples:

```plain text
DETR
DINO
RT-DETR
RF-DETR
LWDETR
```

### 4.3 Distance IoU Loss

Distance IoU adds a penalty for the distance between box centers.

$$
\operatorname{DIoU}(A, B)
=
\operatorname{IoU}(A, B)
-
\frac{
	\rho^2
	\left(
		\hat{c}, c
	\right)
}{
	d^2
}
$$

where:

- $\hat{c}$ is the predicted box center.
- $c$ is the ground-truth box center.
- $\rho^2(\hat{c}, c)$ is the squared Euclidean distance between centers.
- $d^2$ is the squared diagonal length of the smallest enclosing box.

The DIoU loss is:

$$
\mathcal{L}_{DIoU}
=
1
-
\operatorname{DIoU}(A, B)
$$

DIoU encourages predicted boxes to move toward the ground-truth box center.

### 4.4 Complete IoU Loss

Complete IoU further adds an aspect-ratio penalty.

$$
\operatorname{CIoU}(A, B)
=
\operatorname{IoU}(A, B)
-
\frac{
	\rho^2
	\left(
		\hat{c}, c
	\right)
}{
	d^2
}
-
\alpha v
$$

The aspect-ratio term is:

$$
v
=
\frac{4}{\pi^2}
\left(
	\arctan
	\frac{w}{h}
	-
	\arctan
	\frac{\hat{w}}{\hat{h}}
\right)^2
$$

The weighting term is:

$$
\alpha
=
\frac{
	v
}{
	1-\operatorname{IoU}(A,B)+v
}
$$

The CIoU loss is:

$$
\mathcal{L}_{CIoU}
=
1
-
\operatorname{CIoU}(A, B)
$$

CIoU considers:

```plain text
box overlap
center distance
aspect ratio
```

It is commonly used in YOLO-style detectors.

---

## 5. DETR-Style Object Detection Loss

DETR-style models use Hungarian matching to assign predicted object queries to ground-truth objects.

The model outputs a fixed number of queries, while the image contains a variable number of objects.

The Hungarian matcher solves:

$$
\hat{\sigma}
=
\arg\min_{\sigma}
\sum_{j=1}^{M}
\mathcal{C}_{match}
\left(
	\hat{y}_{\sigma(j)}, y_j
\right)
$$

The matching cost is usually:

$$
\mathcal{C}_{match}
=
\lambda_{cls}
\mathcal{C}_{cls}
+
\lambda_{L1}
\mathcal{C}_{L1}
+
\lambda_{giou}
\mathcal{C}_{giou}
$$

After matching, the detection loss is:

$$
\mathcal{L}_{det}
=
\lambda_{cls}
\mathcal{L}_{cls}
+
\lambda_{box}
\mathcal{L}_{box}
+
\lambda_{giou}
\mathcal{L}_{giou}
$$

where:

- $\mathcal{L}_{cls}$ is classification loss.
- $\mathcal{L}_{box}$ is L1 box loss.
- $\mathcal{L}_{giou}$ is GIoU loss.

Matched queries are trained with class and box losses.

Unmatched queries are trained as no object:

$$
y_i = \varnothing
$$

This is common in:

```plain text
DETR
DINO
RT-DETR
RF-DETR
LWDETR
```

---

## 6. YOLO-Style Detection Loss

YOLO-style detectors usually combine classification loss, localization loss, and objectness loss.

A simplified form is:

$$
\mathcal{L}_{YOLO}
=
\lambda_{box}
\mathcal{L}_{box}
+
\lambda_{obj}
\mathcal{L}_{obj}
+
\lambda_{cls}
\mathcal{L}_{cls}
$$

where:

- $\mathcal{L}_{box}$ trains bounding-box localization.
- $\mathcal{L}_{obj}$ trains whether an object exists at a prediction location.
- $\mathcal{L}_{cls}$ trains object class prediction.

The objectness loss is often binary cross-entropy:

$$
\mathcal{L}_{obj}
=
-
\left[
	o
	\log
	\hat{o}
	+
	(1-o)
	\log
	\left(
		1-\hat{o}
	\right)
\right]
$$

where:

- $o$ is the ground-truth objectness label.
- $\hat{o}$ is the predicted objectness probability.

Modern YOLO models often use IoU-based box losses such as CIoU or related variants.

---

## 7. Segmentation Losses

Segmentation losses are used when the model predicts labels or masks for pixels.

Common segmentation tasks include:

```plain text
semantic segmentation
instance segmentation
medical image segmentation
saliency detection
binary foreground segmentation
```

### 7.1 Pixel-Wise Cross-Entropy Loss

For semantic segmentation, each pixel is treated as a classification problem.

$$
\mathcal{L}_{segCE}
=
-
\frac{1}{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\sum_{c=1}^{C}
	y_{i,j,c}
	\log
	\hat{p}_{i,j,c}
$$

where:

- $H$ and $W$ are image height and width.
- $C$ is the number of classes.
- $y_{i,j,c}$ is the ground-truth label for pixel $(i,j)$.
- $\hat{p}_{i,j,c}$ is the predicted probability for class $c$ at pixel $(i,j)$.

Pixel-wise cross-entropy is standard for semantic segmentation.

### 7.2 Mask Binary Cross-Entropy Loss

For binary mask prediction, each pixel is a binary classification problem.

$$
\mathcal{L}_{maskBCE}
=
-
\frac{1}{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\left[
	y_{i,j}
	\log
	\hat{p}_{i,j}
	+
	(1-y_{i,j})
	\log
	\left(
		1-\hat{p}_{i,j}
	\right)
\right]
$$

where:

- $y_{i,j}$ is the ground-truth binary mask value.
- $\hat{p}_{i,j}$ is the predicted foreground probability.

This is common in instance segmentation and binary segmentation.

### 7.3 Dice Loss

Dice loss measures overlap between predicted and ground-truth masks.

The Dice coefficient is:

$$
\operatorname{Dice}
=
\frac{
	2
	\sum_i
	\hat{y}_i y_i
	+
	\epsilon
}{
	\sum_i
	\hat{y}_i
	+
	\sum_i
	y_i
	+
	\epsilon
}
$$

The Dice loss is:

$$
\mathcal{L}_{Dice}
=
1
-
\operatorname{Dice}
$$

where:

- $\hat{y}_i$ is the predicted mask value at pixel $i$.
- $y_i$ is the ground-truth mask value.
- $\epsilon$ prevents division by zero.

Dice loss is useful when foreground objects are small or class imbalance is severe.

Examples:

```plain text
medical segmentation
binary segmentation
small-object segmentation
```

### 7.4 Jaccard / Mask IoU Loss

The Jaccard index, also called mask IoU, is:

$$
\operatorname{Jaccard}
=
\frac{
	\sum_i
	\hat{y}_i y_i
	+
	\epsilon
}{
	\sum_i
	\hat{y}_i
	+
	\sum_i
	y_i
	-
	\sum_i
	\hat{y}_i y_i
	+
	\epsilon
}
$$

The Jaccard loss is:

$$
\mathcal{L}_{Jaccard}
=
1
-
\operatorname{Jaccard}
$$

This loss directly optimizes mask overlap.

### 7.5 Tversky Loss

The Tversky index generalizes Dice by weighting false positives and false negatives differently.

$$
\operatorname{Tversky}
=
\frac{
	TP + \epsilon
}{
	TP
	+
	\alpha FP
	+
	\beta FN
	+
	\epsilon
}
$$

The Tversky loss is:

$$
\mathcal{L}_{Tversky}
=
1
-
\operatorname{Tversky}
$$

where:

- $TP$ is true positives.
- $FP$ is false positives.
- $FN$ is false negatives.
- $\alpha$ controls the false-positive penalty.
- $\beta$ controls the false-negative penalty.

Tversky loss is useful when false negatives and false positives have different importance.

### 7.6 Focal Tversky Loss

Focal Tversky loss focuses more on hard segmentation examples.

$$
\mathcal{L}_{FT}
=
\left(
	1
	-
	\operatorname{Tversky}
\right)^{\gamma}
$$

where:

- $\gamma$ controls how strongly hard examples are emphasized.

This is common in medical image segmentation where missing the foreground can be more costly.

---

## 8. Pose Estimation Losses

Pose estimation losses are used for keypoint prediction.

Common tasks include:

```plain text
human pose estimation
hand pose estimation
animal pose estimation
facial landmark detection
```

### 8.1 Coordinate Regression Loss

If the model directly predicts keypoint coordinates, L1 or L2 coordinate loss can be used.

For $K$ keypoints:

$$
\mathcal{L}_{coord-L1}
=
\frac{
	1
}{
	N_{valid}
}
\sum_{k=1}^{K}
	m_k
	\left\lVert
		\hat{p}_k
		-
		p_k^{*}
	\right\rVert_1
$$

or:

$$
\mathcal{L}_{coord-L2}
=
\frac{
	1
}{
	N_{valid}
}
\sum_{k=1}^{K}
	m_k
	\left\lVert
		\hat{p}_k
		-
		p_k^{*}
	\right\rVert_2^2
$$

where:

- $\hat{p}_k$ is the predicted keypoint coordinate.
- $p_k^{*}$ is the ground-truth keypoint coordinate.
- $m_k$ is the visibility mask.
- $N_{valid} = sum_{k=1}^{K} m_k$.

Coordinate regression is simple, but it often gives lower spatial precision than heatmap-based methods.

### 8.2 Gaussian Heatmap MSE Loss

Heatmap-based pose models predict one heatmap per keypoint.

For keypoint $k$, the Gaussian target heatmap is:

$$
H_k^{*}(u,v)
=
\exp
\left(
	-
	\frac{
		(u-u_k^{*})^2
		+
		(v-v_k^{*})^2
	}{
		2\sigma^2
	}
\right)
$$

where:

- $(u_k^{*}, v_k^{*})$ is the ground-truth keypoint location in heatmap coordinates.
- $\sigma$ controls the Gaussian spread.

The heatmap MSE loss is:

$$
\mathcal{L}_{hm}
=
\frac{
	1
}{
	N_{valid}HW
}
\sum_{k=1}^{K}
\sum_{v=0}^{H-1}
\sum_{u=0}^{W-1}
	m_k
	\left(
		\hat{H}_k(u,v)
		-
		H_k^{*}(u,v)
	\right)^2
$$

where:

- $\hat{H}_k$ is the predicted heatmap.
- $H_k^{*}$ is the target heatmap.
- $m_k$ is the visibility mask.
- $H$ and $W$ are heatmap height and width.

This is one of the most common losses for heatmap-based pose estimation.

### 8.3 OKS Loss

Object Keypoint Similarity, or OKS, is the COCO-style similarity score for keypoint localization.

For keypoint $k$:

$$
OKS_k
=
\exp
\left(
	-
	\frac{
		d_k^2
	}{
		2s^2\kappa_k^2
	}
\right)
$$

where:

- $d_k^2$ is the squared distance between predicted and ground-truth keypoints.
- $s^2$ is the object area.
- $\kappa_k$ is the keypoint-specific tolerance.

The OKS loss is:

$$
\mathcal{L}_{OKS}
=
\frac{
	\sum_{k=1}^{K}
	m_k
	\left(
		1 - OKS_k
	\right)
}{
	\sum_{k=1}^{K}
	m_k
}
$$

OKS loss is useful when the training objective should be closer to the COCO keypoint evaluation metric.

For a heatmap model, OKS loss usually requires differentiable coordinate decoding such as soft-argmax.

---

## 9. Metric Learning Losses

Metric learning losses train embeddings so that similar images are close and dissimilar images are far apart.

Common tasks include:

```plain text
face recognition
person re-identification
image retrieval
self-supervised learning
contrastive learning
```

### 9.1 Contrastive Loss

Contrastive loss uses pairs of samples.

$$
\mathcal{L}_{contrastive}
=
	y
	d^2
	+
	(1-y)
	\max
	\left(
		0,
		m-d
	\right)^2
$$

where:

- $y=1$ means the pair is similar.
- $y=0$ means the pair is dissimilar.
- $d$ is the distance between embeddings.
- $m$ is the margin.

Similar pairs are pulled together, while dissimilar pairs are pushed apart by at least margin $m$.

### 9.2 Triplet Loss

Triplet loss uses three samples:

```plain text
anchor
positive
negative
```

The loss is:

$$
\mathcal{L}_{triplet}
=
\max
\left(
	0,
	d(a,p)
	-
	d(a,n)
	+
	m
\right)
$$

where:

- $a$ is the anchor embedding.
- $p$ is the positive embedding.
- $n$ is the negative embedding.
- $d(a,p)$ is the anchor-positive distance.
- $d(a,n)$ is the anchor-negative distance.
- $m$ is the margin.

The goal is:

$$
d(a,p) + m < d(a,n)
$$

This means the anchor should be closer to the positive sample than to the negative sample.

### 9.3 InfoNCE Loss

InfoNCE is common in contrastive self-supervised learning.

For a query $q$, positive key $k^+$, and negative keys $k_i$:

$$
\mathcal{L}_{InfoNCE}
=
-
\log
\frac{
	\exp
	\left(
		q \cdot k^+ / \tau
	\right)
}{
	\exp
	\left(
		q \cdot k^+ / \tau
	\right)
	+
	\sum_{i=1}^{N}
	\exp
	\left(
		q \cdot k_i / \tau
	\right)
}
$$

where:

- $\tau$ is the temperature.
- $q \cdot k$ measures similarity.
- $k^+$ is the positive key.
- $k_i$ are negative keys.

InfoNCE is used in contrastive learning methods.

Examples:

```plain text
SimCLR
MoCo
CLIP-style contrastive training
```

---

## 10. Image Restoration and Generation Losses

These losses are used when the model outputs an image.

Common tasks include:

```plain text
super-resolution
denoising
deblurring
image generation
style transfer
image translation
```

### 10.1 Pixel L1 Loss

Pixel L1 loss compares predicted and target images pixel by pixel.

$$
\mathcal{L}_{pixel-L1}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left|
	\hat{x}_i
	-
	x_i
\right|
$$

where:

- $\hat{x}_i$ is the predicted pixel value.
- $x_i$ is the target pixel value.
- $N$ is the number of pixels.

L1 loss often produces sharper results than L2 loss in image restoration.

### 10.2 Pixel L2 Loss

Pixel L2 loss is:

$$
\mathcal{L}_{pixel-L2}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
	\hat{x}_i
	-
	x_i
\right)^2
$$

L2 loss strongly penalizes large pixel errors, but it can produce blurry results because it encourages averaging.

### 10.3 Perceptual Loss

Perceptual loss compares high-level features extracted by a pretrained network.

$$
\mathcal{L}_{perc}
=
\sum_{l}
\left\lVert
	\phi_l(\hat{x})
	-
	\phi_l(x)
\right\rVert_2^2
$$

where:

- $\phi_l(\cdot)$ is the feature map from layer $l$ of a pretrained network.
- $\hat{x}$ is the generated image.
- $x$ is the target image.

Perceptual loss encourages generated images to be visually similar at the feature level, not only at the pixel level.

It is commonly used in:

```plain text
super-resolution
style transfer
image generation
```

### 10.4 Style Loss

Style loss compares Gram matrices of feature maps.

For feature map $phi_l(x)$, the Gram matrix is:

$$
G_l(x)
=
\phi_l(x)
\phi_l(x)^T
$$

The style loss is:

$$
\mathcal{L}_{style}
=
\sum_l
\left\lVert
	G_l(\hat{x})
	-
	G_l(x)
\right\rVert_F^2
$$

where:

- $G_l(\hat{x})$ is the Gram matrix of the generated image.
- $G_l(x)$ is the Gram matrix of the style or reference image.
- $\left\lVert \cdot \right\rVert_F$ is the Frobenius norm.

Style loss is commonly used in neural style transfer.

### 10.5 Adversarial Loss

Adversarial loss is used in GANs.

The discriminator loss is:

$$
\mathcal{L}_{D}
=
-
\mathbb{E}_{x \sim p_{data}}
\left[
	\log D(x)
\right]
-
\mathbb{E}_{z \sim p_z}
\left[
	\log
	\left(
		1-D(G(z))
	\right)
\right]
$$

The generator loss is:

$$
\mathcal{L}_{G}
=
-
\mathbb{E}_{z \sim p_z}
\left[
	\log D(G(z))
\right]
$$

where:

- $D(x)$ is the discriminator output.
- $G(z)$ is the generated image.
- $z$ is random noise.
- $p_{data}$ is the real data distribution.

Adversarial loss encourages generated images to look realistic.

---

## 11. Summary Table

| Task | Common Loss Functions |
| --- | --- |
| Image classification | Cross-entropy, label smoothing, focal loss |
| Multi-label classification | Binary cross-entropy |
| Dense object detection | Focal loss, BCE, IoU-based box losses |
| DETR-style object detection | Cross-entropy + L1 + GIoU |
| YOLO-style object detection | Box loss + objectness loss + classification loss |
| Box regression | L1, L2, Smooth L1, IoU, GIoU, DIoU, CIoU |
| Semantic segmentation | Pixel-wise cross-entropy, Dice loss, Jaccard loss |
| Instance segmentation | Mask BCE, Dice loss |
| Medical segmentation | Dice loss, Tversky loss, Focal Tversky loss |
| Pose estimation | Coordinate L1/L2, heatmap MSE, OKS loss |
| Metric learning | Contrastive loss, triplet loss, InfoNCE |
| Image restoration | Pixel L1, pixel L2, perceptual loss |
| Image generation | Perceptual loss, style loss, adversarial loss |

---

## 12. Practical Selection Guide

For image classification:

$$
\mathcal{L}
=
\mathcal{L}_{CE}
$$

For imbalanced classification:

$$
\mathcal{L}
=
\mathcal{L}_{focal}
$$

For DETR-style object detection:

$$
\mathcal{L}
=
\lambda_{cls}
\mathcal{L}_{cls}
+
\lambda_{box}
\mathcal{L}_{box}
+
\lambda_{giou}
\mathcal{L}_{giou}
$$

For YOLO-style object detection:

$$
\mathcal{L}
=
\lambda_{box}
\mathcal{L}_{box}
+
\lambda_{obj}
\mathcal{L}_{obj}
+
\lambda_{cls}
\mathcal{L}_{cls}
$$

For semantic segmentation:

$$
\mathcal{L}
=
\mathcal{L}_{CE}
+
\lambda_{dice}
\mathcal{L}_{Dice}
$$

For heatmap-based pose estimation:

$$
\mathcal{L}
=
\mathcal{L}_{hm}
+
\lambda_{OKS}
\mathcal{L}_{OKS}
$$

For image restoration:

$$
\mathcal{L}
=
\mathcal{L}_{pixel}
+
\lambda_{perc}
\mathcal{L}_{perc}
$$

For image generation:

$$
\mathcal{L}
=
\lambda_{pixel}
\mathcal{L}_{pixel}
+
\lambda_{perc}
\mathcal{L}_{perc}
+
\lambda_{adv}
\mathcal{L}_{adv}
$$

---

## 13. Key Takeaway

The best loss function depends on what the model predicts.

If the model outputs class probabilities, use classification losses.

If the model outputs bounding boxes, use coordinate losses and IoU-based losses.

If the model outputs masks, use pixel-wise losses and overlap-based losses.

If the model outputs keypoint heatmaps, use heatmap losses.

If the model outputs keypoint coordinates, use coordinate losses or OKS-style losses.

If the model outputs embeddings, use metric learning losses.

If the model outputs images, use pixel, perceptual, style, or adversarial losses.