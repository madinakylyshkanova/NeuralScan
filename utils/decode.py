def decode_prediction(preds, alphabet):
    preds = preds.softmax(2)
    preds = preds.argmax(2)  # (T, B)
    preds = preds.permute(1, 0)  # (B, T)

    results = []
    for seq in preds:
        prev = -1
        text = ""

        for i in seq:
            i = i.item()
            if i != prev and i != 0:
                text += alphabet[i - 1]
            prev = i

        results.append(text)

    return results