import numpy as np

def compute_coverage(accepted_mask):
    """
    Coverage = fraction of samples that are accepted
    
    accepted_mask: boolean array of shape (N,)
    returns: float
    """
    accepted_mask = np.asarray(accepted_mask, dtype=bool)
    return accepted_mask.mean()

def compute_selective_risk(correctness, accepted_mask):
    """
    Selective risk = error rate among accepted samples.
    
    correctness: array of 0/1 values
    1 means correct prediction
    0 means wrong prediction
    accepted_mask: boolean array of shape (N,)
    
    returns: float
    """
    correctness = np.asarray(correctness)
    accepted_mask = np.asarray(accepted_mask, dtype=bool)

    accepted_correctness = correctness[accepted_mask]

    if len(accepted_correctness) == 0:
        return 0.0
    
    return 1.0 - accepted_correctness.mean()

def evaluate_at_threshold(confidence_scores, correctness, threshold):
    """
    Accept samples whose confidence is >= threshold,
    then compute coverage and selective risk
    
    Confidence_scores: array of floats
    Correctness: arry of 0/1 values
    threshold: float
    
    returns: 
    accepted_mask, coverage, risk
    """
    confidence_scores = np.asarray(confidence_scores)
    correctness = np.asarray(correctness)

    accepted_mask = confidence_scores >=threshold
    coverage = compute_coverage(accepted_mask)
    risk = compute_selective_risk(correctness, accepted_mask)

    return accepted_mask, coverage, risk

def compute_aurc(confidence_scores, correctness, num_thresholds=100):
    """
    Compute AURC = area under the risk-coverage curve
    using a simple threshold sweep
    
    confidence_scores: array of floats
    correctness: array of 0/1 values
    num_thresholds: int

    returns:
    aurc, covrages, risks, thresholds
    """
    confidence_scores = np.asarray(confidence_scores)
    correctness = np.asarray(correctness)

    thresholds = np.linspace(
        confidence_scores.min(),
        confidence_scores.max(),
        num_thresholds
    )

    coverages = []
    risks = []

    for t in thresholds:
        _, coverage, risk = evaluate_at_threshold(confidence_scores, correctness, t)
        coverages.append(coverage)
        risks.append(risk)

    coverages = np.asarray(coverages)
    risks = np.asarray(risks)

    sort_idx = np.argsort(coverages)
    coverages = coverages[sort_idx]
    risks = risks[sort_idx]
    thresholds = thresholds[sort_idx]

    aurc = np.trapezoid(risks, coverages)

    return aurc, coverages, risks, thresholds
