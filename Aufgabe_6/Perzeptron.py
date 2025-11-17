import numpy as np

def dataset(m):

    # Zufällige Punkte
    X = np.random.uniform(-1, 1, size=(m, 2))

    # Entscheidungslinie
    p1 = np.random.uniform(-1, 1, size=2)
    p2 = np.random.uniform(-1, 1, size=2)

    # Richtungsvektor
    direction = p2 - p1

    # Normalenvektor
    normal = np.array([direction[1], -direction[0]])

    # Bias
    bias = -np.dot(normal, p1)

    # Labels berechnen
    y = np.sign(X @ normal + bias)
    y[y == 0] = 1

    # X um Bias-Dimension erweitern (für w0)
    X_ext = np.hstack([np.ones((m, 1)), X])

    return X_ext, y


def train(X, y, alpha=1):

    w = np.zeros(X.shape[1])
    steps = 0

    while True:
        preds = np.sign(X @ w)
        preds[preds == 0] = 1
        misclassified = np.where(preds != y)[0]

        if len(misclassified) == 0:
            break

        # Fehlerpunkt auswählen (zufällig)
        i = np.random.choice(misclassified)
        w = w + alpha * (y[i] - preds[i]) * X[i]
        steps += 1

    return steps


def run_experiment(m, runs=1000, alpha=1):
    total_steps = 0

    for _ in range(runs):
        X, y = dataset(m)
        steps = train(X, y, alpha)
        total_steps += steps

    return total_steps / runs


if __name__ == "__main__":
    # m = 10, a = 1
    steps_10_a1 = run_experiment(m=10, runs=1000, alpha=1)
    print("Durchschnittliche Schritte für m=10, α=1:", steps_10_a1) 
    # m = 100, a = 1
    steps_100_a1 = run_experiment(m=100, runs=1000, alpha=1)
    print("Durchschnittliche Schritte für m=100, α=1:", steps_100_a1) 
    # m = 1000, a = 1
    steps_1000_a1 = run_experiment(m=1000, runs=1000, alpha=1)
    print("Durchschnittliche Schritte für m=1000, α=1:", steps_1000_a1) 
    # m = 10, a = 0.1
    steps_10_a01 = run_experiment(m=10, runs=1000, alpha=0.1)
    print("Durchschnittliche Schritte für m=10, α=0.1:", steps_10_a01) 
    # m = 100, a = 0.1
    steps_100_a01 = run_experiment(m=100, runs=1000, alpha=0.1)
    print("Durchschnittliche Schritte für m=100, α=0.1:", steps_100_a01) 
    # m = 1000, a = 0.1
    steps_1000_a01 = run_experiment(m=1000, runs=1000, alpha=0.1)
    print("Durchschnittliche Schritte für m=100, α=0.1:", steps_1000_a01) 