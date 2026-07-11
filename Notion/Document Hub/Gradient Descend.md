---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-05-31T14:57:00
Status: Done
Last updated time: 2026-05-31T18:11:00
Last edited by: Heaven Chen
Category:
  - ML
---
Below is the core difference:

- **Vanilla Gradient Descent** updates model parameters using the gradient computed from the **entire dataset**.
- **SGD** updates parameters using the gradient from **one sample or one mini-batch**.
- **Adam** is an adaptive optimizer that keeps track of both the **average gradient direction** and the **average squared gradient size**, so each parameter gets its own adjusted learning rate.

---

# 1. What optimization is doing

Suppose your model has parameters:

$$
\theta
$$

For example, in a neural network, $\theta$ includes all weights and biases.

You also have a loss function:

$$
L(\theta)
$$

The goal of training is:

$$
\min_{\theta} L(\theta)
$$

The gradient tells us the direction in which the loss increases fastest:

$$
\nabla_{\theta} L(\theta)
$$

So if we want to reduce the loss, we move in the opposite direction:

$$
\theta \leftarrow \theta - \eta \, \nabla_{\theta} L(\theta)
$$

where $\eta$ is the **learning rate**.

The learning rate controls how large each update step is.

---

# 2. Vanilla Gradient Descent

Vanilla gradient descent is also called **batch gradient descent**.

It computes the loss over the entire training dataset before making one update.

Suppose your dataset is:

$$
\{(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)\}
$$

The empirical loss is usually:

$$
L(\theta) = \frac{1}{N} \sum_{i=1}^{N} L_i(\theta)
$$

where $L_i(\theta)$ is the loss on one training example.

The vanilla gradient descent update is:

$$
\theta_{t+1} = \theta_t - \eta \, \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} L_i(\theta_t)
$$

So every update uses the average gradient over the whole dataset.

---

## Intuition

Imagine you are standing on a mountain and want to walk downhill.

Vanilla gradient descent looks at the entire landscape carefully before taking one step. Because it uses all data points, its gradient estimate is accurate and stable.

But this is expensive.

If the dataset has one million examples, one parameter update requires computing gradients for all one million examples.

---

## Advantages

Vanilla gradient descent is stable because the gradient is computed from the full dataset.

The update direction is less noisy.

It is conceptually simple and works well for small datasets or convex problems.

---

## Disadvantages

It is slow for large datasets.

It requires loading or processing the entire dataset for every update.

It may get stuck or move slowly in complicated non-convex landscapes, especially in deep learning.

---

# 3. Stochastic Gradient Descent, SGD

SGD modifies vanilla gradient descent by using only a small part of the dataset for each update.

Strictly speaking, **SGD** uses one sample at a time:

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_{\theta} L_i(\theta_t)
$$

where $i$ is one randomly selected training example.

In practice, people often say “SGD” when they mean **mini-batch SGD**.

Mini-batch SGD uses a batch of examples, such as 32, 64, 128, or 256 samples.

If the mini-batch is $B_t$, then:

$$
\theta_{t+1} = \theta_t - \eta \, \frac{1}{|B_t|} \sum_{i \in B_t} \nabla_{\theta} L_i(\theta_t)
$$

---

## Intuition

Instead of looking at the entire landscape before every step, SGD looks at a small random portion of the landscape and makes a rough guess about the downhill direction.

This guess is noisy, but it is much cheaper.

So SGD makes many small, noisy updates.

---

## Why SGD works

The mini-batch gradient is a noisy estimate of the full gradient.

The full gradient is:

$$
\nabla_{\theta} L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} L_i(\theta)
$$

The mini-batch gradient is:

$$
g_t = \frac{1}{|B_t|} \sum_{i \in B_t} \nabla_{\theta} L_i(\theta_t)
$$

Although $g_t$ is noisy, it is usually an unbiased estimate of the full gradient:

$$
\mathbb{E}[g_t] = \nabla_{\theta} L(\theta_t)
$$

So on average, SGD moves in the right direction.

---

## Advantages

SGD is much faster per update than vanilla gradient descent.

It works well for large datasets.

The noise can help the model escape shallow local minima or saddle points.

It is memory-efficient because it only needs a mini-batch at a time.

---

## Disadvantages

The update direction is noisy.

The loss may fluctuate instead of decreasing smoothly.

It is sensitive to the learning rate.

If the learning rate is too large, training may diverge.

If the learning rate is too small, training becomes slow.

---

# 4. SGD with Momentum

Before Adam, one important improvement to SGD was **momentum**.

Plain SGD only uses the current gradient.

Momentum remembers the previous update direction.

The update is:

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

where:

$$
g_t = \nabla_{\theta} L(\theta_t)
$$

and $v_t$ is a moving average of gradients.

---

## Intuition

Momentum is like a ball rolling downhill.

If gradients keep pointing in a similar direction, momentum accelerates movement in that direction.

If gradients oscillate back and forth, momentum smooths out the oscillation.

This is especially useful in narrow valleys where SGD zigzags.

---

# 5. Adam

Adam stands for **Adaptive Moment Estimation**.

Adam combines two ideas:

1. **Momentum**, which tracks the moving average of gradients.
2. **Adaptive learning rates**, which scale each parameter update based on recent gradient magnitudes.

Adam keeps two moving averages.

First moment (moving average of gradients):

$$
m_t
$$

Second moment (moving average of squared gradients):

$$
v_t
$$

Given gradient:

$$
g_t = \nabla_{\theta} L(\theta_t)
$$

Adam computes:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Then it uses bias correction:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Finally, the parameter update is:

$$
\theta_{t+1} = \theta_t - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

---

## What does Adam mean intuitively?

Adam asks two questions for each parameter.

**Question 1: Which direction has this parameter usually wanted to move?**

That is handled by $m_t$, which is similar to momentum.

**Question 2: How large have the gradients for this parameter usually been?**

That is handled by $v_t$.

If a parameter usually has very large gradients, Adam reduces its effective step size.

If a parameter usually has very small gradients, Adam allows relatively larger movement.

So Adam gives each parameter its own adaptive learning rate.

---

# 6. Why Adam divides by the square root of $v_t$

The term:

$$
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

means Adam normalizes the update by the recent magnitude of gradients.

For a parameter with large gradients, $\sqrt{\hat{v}_t}$ is large, so the update becomes smaller.

For a parameter with small gradients, $\sqrt{\hat{v}_t}$ is small, so the update becomes relatively larger.

This is useful when different parameters have gradients on very different scales.

That often happens in deep learning.

---

# 7. Why Adam needs bias correction

At the beginning of training:

$$
m_0 = 0
$$

$$
v_0 = 0
$$

So the moving averages are biased toward zero in the early steps.

For example:

$$
m_1 = \beta_1 m_0 + (1 - \beta_1)g_1
$$

Since $m_0 = 0$, this becomes:

$$
m_1 = (1 - \beta_1)g_1
$$

If $\beta_1 = 0.9$, then:

$$
m_1 = 0.1 g_1
$$

That is much smaller than the actual gradient.

So Adam corrects this using:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

and:

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

This makes early training more stable.

---

# 8. Comparing vanilla GD, SGD, and Adam

| Optimizer | Gradient source | Main idea | Strength | Weakness |
| --- | --- | --- | --- | --- |
| Vanilla Gradient Descent | Entire dataset | Use exact full gradient | Stable direction | Very slow for large datasets |
| SGD | One sample or mini-batch | Use noisy gradient estimate | Fast and scalable | Noisy, sensitive to learning rate |
| SGD + Momentum | Mini-batch + moving average | Smooth gradient direction | Reduces zigzagging | Still needs careful tuning |
| Adam | Mini-batch + adaptive moments | Momentum + per-parameter scaling | Fast, robust default | Can generalize worse than SGD in some cases |

---

# 9. Concrete example

Suppose we are training a simple linear regression model:

$$
\hat{y} = wx + b
$$

The parameters are:

$$
\theta = \{w, b\}
$$

The loss could be mean squared error:

$$
L(w, b) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2
$$

where:

$$
\hat{y}_i = wx_i + b
$$

---

## Vanilla Gradient Descent

Use all training examples to compute the gradient:

$$
\frac{\partial L}{\partial w}
$$

$$
\frac{\partial L}{\partial b}
$$

Then update:

$$
w \leftarrow w - \eta \, \frac{\partial L}{\partial w}
$$

$$
b \leftarrow b - \eta \, \frac{\partial L}{\partial b}
$$

One update uses the whole dataset.

---

## SGD

Randomly choose one example or mini-batch.

Compute the gradient from only that subset.

Then update:

$$
w \leftarrow w - \eta \, \frac{\partial L_B}{\partial w}
$$

$$
b \leftarrow b - \eta \, \frac{\partial L_B}{\partial b}
$$

One update is cheaper, but noisier.

---

## Adam

Adam still computes mini-batch gradients, but it does not directly use them for the update.

Instead, for $w$ and $b$, it tracks $m_t$ (the moving average of gradients) and $v_t$ (the moving average of squared gradients).

Then it updates each parameter with an individually scaled step.

So $w$ and $b$ may effectively receive different learning rates.

---

# 10. Visual intuition

Imagine the loss surface is shaped like a long narrow valley.

Vanilla gradient descent may move smoothly but slowly.

SGD may bounce around because each mini-batch gives a slightly different direction.

SGD with momentum reduces bouncing by remembering the general direction.

Adam goes further by saying:

> “Some parameters are changing too aggressively, so I will slow them down. Some parameters are barely moving, so I will let them move more.”

That is why Adam often works well out of the box.

---

# 11. Why people often start with Adam

In deep learning, Adam is popular because:

- It usually converges quickly.
- It is less sensitive to the initial learning rate than plain SGD.
- It handles sparse gradients well.
- It works well when parameters have different gradient scales.

A common default setting is:

$$
\eta = 0.001
$$

$$
\beta_1 = 0.9
$$

$$
\beta_2 = 0.999
$$

$$
\epsilon = 10^{-8}
$$

---

# 12. Why people still use SGD

Even though Adam is convenient, SGD is still widely used, especially in computer vision.

The reason is that SGD sometimes gives better **generalization**.

That means the model may perform better on unseen test data, even if Adam reaches low training loss faster.

A common pattern is:

- Adam: faster training loss decrease.
- SGD: sometimes better final test accuracy.

This is not always true, but it is a common practical observation.

---

# 13. Main difference in one sentence each

**Vanilla gradient descent**: uses the entire dataset to compute one accurate but expensive update.

**SGD**: uses a random sample or mini-batch to compute a cheap but noisy update.

**Adam**: uses mini-batch gradients like SGD, but adjusts each parameter using momentum and adaptive learning-rate scaling.

---

# 14. Simple PyTorch examples

## Vanilla-like full-batch gradient descent

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Use the entire dataset at once
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()
```

This is only practical if the whole dataset fits in memory.

---

## Mini-batch SGD

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()
```

Here, each update uses one mini-batch.

---

## SGD with momentum

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()
```

This is SGD plus memory of previous gradient directions.

---

## Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)

for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()
```

Adam uses the gradient from each mini-batch, but internally keeps moving averages of gradients and squared gradients.

---

# 15. Practical advice

For most deep learning experiments, start with:

```python
torch.optim.Adam(model.parameters(), lr=1e-3)
```

For transformer models or large pretrained models, people often use:

```python
torch.optim.AdamW(model.parameters(), lr=1e-5)
```

or:

```python
torch.optim.AdamW(model.parameters(), lr=2e-5)
```

For computer vision models trained from scratch, SGD with momentum is often strong:

```python
torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)
```

But for fine-tuning pretrained models, AdamW is usually a safer default.

---

# 16. Adam vs AdamW

One important detail: in modern deep learning, people often prefer **AdamW** over Adam.

AdamW is Adam with decoupled weight decay.

In classic Adam, weight decay can interact strangely with the adaptive learning rate. AdamW separates weight decay from the gradient update, making regularization behave more cleanly.

For neural networks today, especially transformers, AdamW is usually preferred over Adam.

---

# 17. Summary

Vanilla gradient descent is the cleanest theoretical version: compute the full gradient and update once.

SGD makes training scalable by using random mini-batches.

Adam improves SGD by adding momentum and adaptive per-parameter scaling.

A simple way to remember them:

$$
\text{GD} = \text{full-data gradient}
$$

$$
\text{SGD} = \text{mini-batch noisy gradient}
$$

$$
\text{Adam} = \text{SGD} + \text{momentum} + \text{adaptive scaling}
$$

In modern neural network training, pure vanilla gradient descent is rarely used. Mini-batch SGD, Adam, and AdamW are much more common.