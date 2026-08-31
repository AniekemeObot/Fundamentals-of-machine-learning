# Fundamentals-of-machine-learning

# Understanding Machine Learning: Theory and a Worked Example

## What Machine Learning Actually Is

Traditional programming: you write explicit rules — `if X then Y` — and the computer follows your logic exactly.

Machine learning flips this. You give the computer examples (data) and let it find the pattern or rule itself, rather than hand-coding the rule yourself. You supply inputs and known correct outputs; the algorithm adjusts internal parameters until its own predictions match those outputs closely enough then it applies that learned pattern to new, unseen inputs.

## Three Core Types

**Supervised learning**  you have labeled examples (input → correct answer). E.g., past house sizes and their actual prices → predict the price of a new house. Most beginner ML lives here.

**Unsupervised learning** — no labels, just data; the algorithm finds structure or groupings on its own (e.g., clustering similar customers).

**Reinforcement learning** — an agent learns by trial and error, receiving rewards or penalties (e.g., game-playing AI).

## The Mechanics, Stripped Down

1. Collect data (inputs + known outputs)
2. Choose a model — a mathematical function with adjustable parameters (can be as simple as `y = mx + b`)
3. Feed data through it, measure how wrong its predictions are (the "loss")
4. Adjust parameters to reduce that error — this is what "training" means
5. Repeat until error is acceptably low
6. Test on new data it hasn't seen, to check it actually generalized rather than just memorized

## The Most Important Beginner Insight: Overfitting

A model that fits its training data perfectly but performs badly on new, unseen data has **overfit** — it memorized noise instead of learning the real underlying pattern. This is the failure mode almost everyone encounters first, and it's why the train/test split below matters: without it, you can't tell the difference between a model that actually learned something and one that just memorized the answers.

## Worked Example: Linear Regression from Scratch

The clearest way to see this whole loop in action, with minimal code, is linear regression — predicting one number from another using the `y = mx + b` idea, except the computer *finds* `m` and `b` from data rather than a person guessing them.

The script below implements this using gradient descent (not a library's one-line `.fit()` call), so each step of the theory above — data, model, loss, training loop, evaluation — is visible and traceable in the code itself.

```python
"""
Linear Regression from Scratch
--------------------------------
Demonstrates the core mechanics of machine learning: a model learns
parameters (m, b) from data by iteratively reducing prediction error,
rather than being explicitly programmed with a rule.

Model: y = m*x + b
Goal: find m and b that best fit the data, using gradient descent.
"""

import numpy as np

# 1. DATA
# Synthetic example: hours studied -> exam score.
# In real use, this would come from a CSV, database, or API call.
np.random.seed(42)
hours_studied = np.linspace(1, 10, 30)
true_m, true_b = 8.5, 20
noise = np.random.normal(0, 5, size=hours_studied.shape)
exam_scores = true_m * hours_studied + true_b + noise

# 2. TRAIN/TEST SPLIT
# Held-out data checks whether the model generalizes, not just memorizes.
split_index = int(0.8 * len(hours_studied))
x_train, x_test = hours_studied[:split_index], hours_studied[split_index:]
y_train, y_test = exam_scores[:split_index], exam_scores[split_index:]

# 3. MODEL + LOSS
# Model: a straight line with two learnable parameters, m and b.
# Loss: mean squared error -- how far predictions are from actual values.
def predict(x, m, b):
    return m * x + b

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 4. TRAINING LOOP (gradient descent)
# Start with random guesses, then nudge m and b in the direction
# that reduces the loss, repeated many times.
m, b = 0.0, 0.0
learning_rate = 0.01
epochs = 1000
n = len(x_train)

for epoch in range(epochs):
    y_pred = predict(x_train, m, b)
    error = y_pred - y_train

    # Gradients: how much the loss changes per unit change in m and b.
    grad_m = (2 / n) * np.dot(error, x_train)
    grad_b = (2 / n) * np.sum(error)

    m -= learning_rate * grad_m
    b -= learning_rate * grad_b

    if epoch % 200 == 0:
        loss = mean_squared_error(y_train, y_pred)
        print(f"Epoch {epoch:4d} | loss: {loss:8.2f} | m: {m:.2f} | b: {b:.2f}")

# 5. EVALUATION ON UNSEEN DATA
# This is the real test: did the model learn the pattern, or just
# memorize the training examples?
test_predictions = predict(x_test, m, b)
test_loss = mean_squared_error(y_test, test_predictions)

print(f"\nLearned model: y = {m:.2f}x + {b:.2f}")
print(f"(True underlying relationship was: y = {true_m}x + {true_b})")
print(f"Test set loss (MSE): {test_loss:.2f}")

# 6. USE THE MODEL
new_hours = 7
predicted_score = predict(new_hours, m, b)
print(f"\nPredicted score for {new_hours} hours studied: {predicted_score:.1f}")
```

## What Running This Shows

Over 1000 epochs, `m` and `b` start at 0 and gradually converge toward the true underlying relationship (`m=8.5, b=20`) used to generate the synthetic data — without ever being told those values directly. The loss (mean squared error) decreases as training progresses, and the test set evaluation confirms the learned relationship generalizes to data the model never saw during training, rather than just memorizing the training examples.