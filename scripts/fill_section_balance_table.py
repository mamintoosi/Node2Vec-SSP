import json, os, sys

def B_from_labels(path):
    """Section balance B = min(|S1|,|S2|) / max(|S1|,|S2|) from saved cluster labels."""
    d = json.load(open(path))
    cl = d["cluster_labels"]
    c0 = sum(1 for x in cl if x == 0)
    c1 = len(cl) - c0
    if c0 == 0 or c1 == 0:
        return None
    return min(c0, c1) / max(c0, c1)

def main():
    base = "results/grid_search"

    d1 = {}
    print("d=1, p=2.0, q=1.0  (primary Node2Vec)")
    for c in range(1, 7):
        p = f"{base}/labels_p2.0_q1.0_d1_course{c}.json"
        b = B_from_labels(p)
        d1[c] = b
        print(f" course {c}: B={b:.4f}")

    d2 = {}
    print("\nd=2, p=1.0, q=1.0  (neutral Node2Vec)")
    for c in range(1, 7):
        p = f"{base}/labels_p1.0_q1.0_d2_course{c}.json"
        b = B_from_labels(p)
        d2[c] = b
        print(f" course {c}: B={b:.4f}")

    print("\nPrimary d=1 avg:", round(sum(d1.values())/len(d1), 4))
    print("Neutral d=2 avg:", round(sum(d2.values())/len(d2), 4))

    # Optional: also print LaTeX-ready rows
    print("\nLaTeX rows (course, B_d1, B_d2):")
    for c in range(1, 7):
        print(f" {c} & {d1[c]:.3f} & {d2[c]:.3f} \\\\")
    print(f" \\textbf{{Avg.}} & {sum(d1.values())/len(d1):.3f} & {sum(d2.values())/len(d2):.3f} \\\\")

if __name__ == "__main__":
    main()
