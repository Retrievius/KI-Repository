import pandas as pd
from collections import Counter

# --------------------------
# Hilfsfunktionen
# --------------------------

def calc3_score(df, attribute, target):
    """Berechnet die Klassifikationsgüte (calc3) für ein Attribut."""
    total = len(df)
    weighted_sum = 0.0
    for value, subset in df.groupby(attribute):
        counts = subset[target].value_counts()
        purity = counts.max() / len(subset)
        weighted_sum += len(subset) * purity
    return weighted_sum / total

def majority_class(df, target):
    """Gibt die häufigste Klasse in einem Subset zurück."""
    return df[target].value_counts().idxmax()

# --------------------------
# CAL3 Entscheidungsbaum
# --------------------------

def cal3(df, features, target, S1=4, S2=0.7):
    """
    Rekursive CAL3-Baumkonstruktion.
    Stoppt, wenn weniger als S1 Elemente oder Güte < S2.
    """
    # Wenn alle Beispiele die gleiche Klasse haben → Leaf
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # Stopbedingungen
    if len(df) < S1:
        return majority_class(df, target)

    # calc3 für alle Features berechnen
    scores = {f: calc3_score(df, f, target) for f in features}
    best_attr = max(scores, key=scores.get)
    best_score = scores[best_attr]

    # Wenn Güte zu klein → kein Split mehr
    if best_score < S2:
        return majority_class(df, target)

    # Teilmengen bilden
    tree = {best_attr: {}}
    for value, subset in df.groupby(best_attr):
        if subset.empty:
            tree[best_attr][value] = majority_class(df, target)
        else:
            remaining = [f for f in features if f != best_attr]
            tree[best_attr][value] = cal3(subset, remaining, target, S1, S2)
    return tree

import math

def entropy(df, target):
    """Berechnet die Entropie einer Datenmenge."""
    counts = df[target].value_counts()
    total = len(df)
    return -sum((count/total) * math.log2(count/total) for count in counts)

def info_gain(df, attribute, target):
    """Berechnet den Informationsgewinn durch Aufteilung nach einem Attribut."""
    total_entropy = entropy(df, target)
    total = len(df)
    weighted_entropy = 0.0
    for value, subset in df.groupby(attribute):
        weighted_entropy += (len(subset)/total) * entropy(subset, target)
    return total_entropy - weighted_entropy

def id3(df, features, target):
    """Rekursive ID3-Entscheidungsbaum-Erstellung."""
    # Wenn alle die gleiche Klasse → Leaf
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # Keine Features mehr → Mehrheitsklasse
    if not features:
        return majority_class(df, target)

    # Bester Split nach Information Gain
    gains = {f: info_gain(df, f, target) for f in features}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}
    for value, subset in df.groupby(best_attr):
        if subset.empty:
            tree[best_attr][value] = majority_class(df, target)
        else:
            remaining = [f for f in features if f != best_attr]
            tree[best_attr][value] = id3(subset, remaining, target)
    return tree

data = [
    ["≥35", "hoch", "Abitur", "O"],
    ["<35", "niedrig", "Master", "O"],
    ["≥35", "hoch", "Bachelor", "M"],
    ["≥35", "niedrig", "Abitur", "M"],
    ["≥35", "hoch", "Master", "O"],
    ["<35", "hoch", "Bachelor", "O"],
    ["<35", "niedrig", "Abitur", "M"]
]

df = pd.DataFrame(data, columns=["Alter", "Einkommen", "Bildung", "Kandidat"])

features = ["Alter", "Einkommen", "Bildung"]
target = "Kandidat"

# CAL3-Baum
cal3_tree = cal3(df, features, target, S1=4, S2=0.7)
print("CAL3 Entscheidungsbaum:")
print(cal3_tree)

# ID3-Baum
id3_tree = id3(df, features, target)
print("\nID3 Entscheidungsbaum:")
print(id3_tree)
