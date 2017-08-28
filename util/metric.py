def categorical_metric(y_test, y_predict, label2idx):

    def idxSeq2label(idxSeq, lookup):
        return list(map(lambda idx: lookup[idx], idxSeq))

    def transform2label(y, lookup):
        y_label = []
        for seq in y:
            y_label.append(idxSeq2label(seq, lookup))
        return y_label

    def f1(label_tags, predict_tags, keyIdx=2):
        label_count = 0
        predict_count = 0
        intersection_count = 0

        for label_tag, predict_tag in zip(label_tags, predict_tags):
            for label, predict in zip(label_tag, predict_tag):
                if label == keyIdx:
                    label_count += 1
                if predict == keyIdx:
                    predict_count += 1
                if label == keyIdx and predict == keyIdx:
                    intersection_count += 1

        p = intersection_count / float(predict_count)
        r = intersection_count / float(label_count)
        f = 2 * p * r / float(p + r)
        return p, r, f


    keyIdx = label2idx['1']

    return f1(y_test, y_predict, keyIdx=keyIdx)


