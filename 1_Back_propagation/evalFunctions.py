import numpy as np


def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    LPred = np.asarray(LPred).ravel()
    LTrue = np.asarray(LTrue).ravel()

    acc = np.mean(LPred == LTrue)
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    LPred = np.asarray(LPred).ravel()
    LTrue = np.asarray(LTrue).ravel()

    # All class labels that occur in either vector, sorted
    classes = np.unique(np.concatenate((LTrue, LPred)))
    nClasses = classes.shape[0]

    # Map the labels to indices 0, 1, ..., nClasses-1
    predInd = np.searchsorted(classes, LPred)
    trueInd = np.searchsorted(classes, LTrue)

    # Count occurrences of each (predicted, actual) pair
    cM = np.zeros((nClasses, nClasses), dtype=int)
    np.add.at(cM, (predInd, trueInd), 1)
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    cM = np.asarray(cM)

    # Correct predictions sit on the diagonal
    acc = np.trace(cM) / np.sum(cM)
    # ============================================

    return acc