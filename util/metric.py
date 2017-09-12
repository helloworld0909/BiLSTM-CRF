def binary_metric(y_test, y_predict, label2idx):

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

def categorical_metric(y_test, y_predict):

    def acc(label_tags, predict_tags):
        total_count = 0
        correct_count = 0
        for label_tag, predict_tag in zip(label_tags, predict_tags):
            for label, predict in zip(label_tag, predict_tag):
                if label != 0:
                    total_count += 1
                    if label == predict:
                        correct_count += 1
        return correct_count / float(total_count)

    return acc(y_test, y_predict)
