import numpy as np
from eval.metrics import evaluate_at_threshold, compute_aurc

#Fake ground truth labels
y_true = np.array([0,1,2,1,0,2,1,0,2,1])

#Fake predicted labels
y_pred = np.array([0,1,1,1,0,2,0,0,2,2])

#Fake confidence scores for each prediction
confidence_scores = np.array([0.95, 0.90, 0.40, 0.80, 0.88, 0.93, 0.55, 0.70, 0.85, 0.45])

#1 = correct, 0 = wrong
correctness = (y_true == y_pred).astype(int)

#Test one threshold
threshold = 0.75
accepted_mask, coverage, risk = evaluate_at_threshold(
    confidence_scores,
    correctness,
    threshold
)

print("Single Threshold Evaluation")
print("Threshold: ", threshold)
print("Accepted mask:", accepted_mask)
print("Coverage:", coverage)
print("selective risk:", risk)


# Compute AURC by sweeping thresholds
aurc, coverags, risks, thresholds = compute_aurc(
    confidence_scores,
    correctness,
    num_thresholds=50
)

print("AURC Evaluation")
print("AURC:", aurc)

print("Sample points from risk-coverage curve:")
for i in range(0, len(coverags), 10):
    print(f"Threshold={thresholds[i]:.3f}, Coverage={coverags[i]:.3f}, Risk ={risks[i]:.3f}")