---

---
# Feature Extraction for Fidgeting Detection

The goal is to convert short sequences of 2-D keypoint trajectories into a fixed-length feature vector that can be used by a classifier to distinguish fidgeting from non-fidgeting behavior.

Assume that each frame contains (K) body keypoints. Since every keypoint has both an (x)-coordinate and a (y)-coordinate, the total number of one-dimensional coordinate trajectories is:

$$
D = 2K
$$

For example, if we use 17 body keypoints:

$$
D = 2 \times 17 = 34
$$

The trajectories can be represented as:

$$
x_1(t), y_1(t), x_2(t), y_2(t), \dots, x_K(t), y_K(t)
$$

Each analysis window contains 100 frames. For every 100-frame window, three types of features are extracted:

1. Fast Fourier Transform (FFT) features
2. Standard deviation (STD) features
3. Mean (MEAN) features

---

## 1. FFT Features: Rhythmic Motion Patterns

Fidgeting often contains repetitive or cyclic motion, such as leg bouncing, finger tapping, hand shaking, or repeated arm movement.

The authors assume that fidgeting-related cyclic movement mostly occurs within the frequency range:

$$
0.5 \text{ Hz} \leq f \leq 2.5 \text{ Hz}
$$

This means the relevant movement repeats approximately once every 0.4 to 2 seconds:

$$
\frac{1}{2.5} = 0.4 \text{ seconds}
$$

$$
\frac{1}{0.5} = 2 \text{ seconds}
$$

For each coordinate trajectory, such as the horizontal movement of the left wrist, an FFT is applied:

$$
x_d(t) \xrightarrow{\text{FFT}} X_d(f)
$$

The FFT converts the trajectory from the time domain into the frequency domain.

Instead of describing where a keypoint was at each frame, the FFT describes how strongly that keypoint moved at different motion frequencies.

For example:

- A stationary hand produces little energy at most frequencies.
- A hand repeatedly moving up and down produces stronger energy at a particular frequency.
- A bouncing leg may produce strong energy between 0.5 Hz and 2.5 Hz.

Because the analysis window always contains 100 frames, the number of FFT frequency bins within the selected range ([0.5, 2.5]) Hz is fixed at 41.

Therefore, for every trajectory, the FFT produces:

$$
41 \text{ frequency values}
$$

Since there are (D = 2K) trajectories, the full FFT representation before pooling would have size:

$$
41 \times D
$$

For 17 keypoints:

$$
41 \times 34 = 1394
$$

However, the authors average FFT magnitudes across all coordinate trajectories for each frequency bin:

$$
\text{FFTFeature}(f_j)
=

\frac{1}{D}
\sum_{d=1}^{D}
|X_d(f_j)|
$$

This produces one value per frequency bin rather than one value per frequency bin per trajectory.

Therefore, the final FFT feature has length:

$$
41
$$

This feature describes the overall strength of rhythmic motion in the body at each frequency between 0.5 Hz and 2.5 Hz.

---

## 2. Standard Deviation Features: Motion Magnitude

For each coordinate trajectory, the standard deviation is calculated over the 100 frames:

$$
\mathrm{STD}_d
=

\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
\left(x_d(t)-\mu_d\right)^2
}
$$

where:

$$
T = 100
$$

and:

$$
\mu_d
=

\frac{1}{T}
\sum_{t=1}^{T}x_d(t)
$$

The standard deviation measures how much each coordinate changes over time.

A high standard deviation means that the coordinate moved substantially during the 100-frame window.

Examples:

- A stationary hand has low STD.
- A tapping finger or moving wrist has high STD.
- A bouncing knee or ankle has high STD.
- A person maintaining the same seated pose has low STD across most body coordinates.

One STD value is retained for every coordinate trajectory:

$$
\mathrm{STD}
\in
\mathbb{R}^{D}
$$

Since:

$$
D = 2K
$$

the STD feature vector has length:

$$
2K
$$

For 17 keypoints:

$$
\mathrm{STD}
\in
\mathbb{R}^{34}
$$

Unlike the pooled FFT feature, STD preserves which individual coordinate moved. This allows the classifier to learn that movement in certain body parts, such as wrists, ankles, knees, or fingers, may be more associated with fidgeting.

---

## 3. Mean Features: Average Position and Pose

For each coordinate trajectory, the mean position across the 100 frames is calculated:

$$
\mathrm{MEAN}_d
=

\frac{1}{T}
\sum_{t=1}^{T}x_d(t)
$$

This also produces one value for every coordinate trajectory:

$$
\mathrm{MEAN}
\in
\mathbb{R}^{D}
$$

Therefore, the MEAN feature vector has length:

$$
2K
$$

For 17 keypoints:

$$
\mathrm{MEAN}
\in
\mathbb{R}^{34}
$$

The MEAN feature captures the average body pose or location of each keypoint during the time window.

For example, it may help distinguish between:

- A hand moving near the face
- A hand moving near the lap
- A foot moving below a chair
- A wrist moving during a broad arm gesture
- A person who is seated versus standing

The MEAN feature is useful because similar rhythmic motion can have different meanings depending on where it happens in the body.

---

## Combined Feature Representation

The final input feature vector is formed by concatenating the FFT, STD, and MEAN features:

$$
\text{Feature}
==============

[
\text{FFT},
\text{STD},
\text{MEAN}
]
$$

The dimensions are:

| Feature Type | Dimension | Main Information Captured |
| --- | --- | --- |
| FFT | (41) | Strength of cyclic motion at frequencies from 0.5 Hz to 2.5 Hz |
| STD | (2K) | Amount of movement for each keypoint coordinate |
| MEAN | (2K) | Average position of each keypoint coordinate |
| **Total** | (41 + 4K) | Combined rhythmic motion, movement magnitude, and pose information |

For (K = 17) keypoints:

$$
41 + 2K + 2K
= 41 + 34 + 34
=
109
$$

Therefore, every 100-frame motion segment is converted into a fixed-length feature vector of:

$$
109
$$

dimensions.

---

## Intuitive Summary

| Feature | Main Question It Answers | Example Interpretation |
| --- | --- | --- |
| FFT | Is the person moving rhythmically at a fidgeting-related frequency? | Repetitive leg bouncing may create strong energy around 1–2 Hz. |
| STD | How much did each body coordinate move? | A highly moving wrist or ankle produces a larger STD value. |
| MEAN | Where was the body part located on average? | Motion near the face may indicate a different behavior from motion near the lap. |

Together, these features allow the model to recognize not only whether movement occurred, but also:

- whether the movement was repetitive,
- how large the movement was, and
- where on the body the movement occurred.