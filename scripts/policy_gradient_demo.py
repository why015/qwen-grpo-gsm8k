import math
import random

random.seed(42)
theta = 0.0
learning_rate = 0.1

def probability(x):
    return 1 / (1 + math.exp(-theta * x))

print(f"训练前：x=1 选1的概率 = {probability(1):.3f}")

for step in range(1, 1001):
    x = random.choice([-1, 1])
    p = probability(x)
    action = int(random.random() < p)
    correct_action = int(x == 1)
    reward = int(action == correct_action)

    gradient = (action - p) * x
    theta += learning_rate * reward * gradient

    if step % 200 == 0:
        print(f"第{step:4d}步：theta={theta:.3f}，x=1 选1的概率={probability(1):.3f}")

print(f"训练后：x=-1 选0的概率 = {1 - probability(-1):.3f}")
