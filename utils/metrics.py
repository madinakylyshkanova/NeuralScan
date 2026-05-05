def cer(pred, gt):
    dp = [[0]*(len(gt)+1) for _ in range(len(pred)+1)]

    for i in range(len(pred)+1):
        dp[i][0] = i
    for j in range(len(gt)+1):
        dp[0][j] = j

    for i in range(1, len(pred)+1):
        for j in range(1, len(gt)+1):
            cost = 0 if pred[i-1] == gt[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[-1][-1] / max(1, len(gt))


def average_cer(preds, gts):
    return sum(cer(p, g) for p, g in zip(preds, gts)) / len(preds)


def word_accuracy(preds, gts):
    return sum(p == g for p, g in zip(preds, gts)) / len(preds)