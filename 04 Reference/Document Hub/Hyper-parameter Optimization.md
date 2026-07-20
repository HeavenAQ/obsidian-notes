---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-07-20T11:29:00
Status: In-Progress
Last updated time: 2026-07-20T11:29:00
Last edited by: Heaven Chen
Category:
  - ML
  - Finetuning
---
## Learning Rate Scheduler

### `StepLR`

- Decrease learning rate by a constant factor
- `gamma`: factor of rate decay
- `step_size`: number of epochs
![[Pasted image 20260720113222.png|521]]
```python
scheduler = optim.lr_scheduler.StepLR(
	optimizer, step_size=10, gamma=0.2
)
scheduler.step()
```

### `ReduceLROnPlateau`
* `factor`: factor of the rate decay
* `patience`: Number of epochs without improvement

![[Pasted image 20260720113852.png]]
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
	optimizer, mode='max', factor=0.2, patience=3,
)
scheduler.step(val_acc)
```

### `CosineAnnealingLR`
* Gradually decrease the learning rate
* smoother transition than the other two schedulers
* `T_max`: Max number of iterations
* `eta_min`: Min learning rate
![[Pasted image 20260720114140.png]]
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
	optimizer, T_max=n_epochs, eta_min=0.0002
)
```
## Architecture

### Number of Layers (depth)

- Affect the ability to learn complex patterns
- **Shallow Layers**
  - require more feature engineering
  - efficient in training
  - hard to learn from complex datasets
- **Deep Layers**
  - extract more complex features
  - higher risks of overfitting (especially for small dataset)
  - longer training time

### Number of Neurons per Layer (width)

- More neurons
  - better results for complex dataset
  - more weights to learn -> memory overhead

## Weight Decay

### L2 Regularization

- discourage large weights and in favor of simpler models

$$
Loss = Error(Y - \hat{Y}) + \lambda \sum^n_{1} w^2_i
$$

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```
