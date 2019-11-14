def div(a, b):
    return 0 if b == 0 else a / b


def f_measure(prec, recall):
    return div(2 * prec * recall, prec + recall)


def macro_f_measure(n, tp, fp, fn, we):
    precision_sum = 0
    recall_sum = 0
    for i in range(n):
        precision_sum += we[i] * div(tp[i], tp[i] + fp[i])
        recall_sum += we[i] * div(tp[i], tp[i] + fn[i])
    return f_measure(recall_sum, precision_sum)


def micro_f_measure(n, tp, fp, fn, we):
    mi_f_measure = 0
    for i in range(n):
        mi_f_measure += we[i] * f_measure(div(tp[i], tp[i] + fp[i]), div(tp[i], tp[i] + fn[i]))
    return mi_f_measure


def count_micro_f_measure(confusion_matrix, n):
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    we = [0.0] * n
    exp_num_sum = 0
    for i in range(n):
        row = confusion_matrix[i]
        for j in range(n):
            exp_num_sum += row[j]
            if j != i:
                fp[i] += row[j]
                fn[j] += row[j]
            else:
                tp[i] += row[j]
    for i in range(n):
        we[i] = div(tp[i] + fp[i], exp_num_sum)
    mi_f_measure = micro_f_measure(n, tp, fp, fn, we)
    return mi_f_measure


def solve():
    n = int(input())
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    we = [0.0] * n
    exp_num_sum = 0
    for i in range(n):
        row = list(map(int, input().split()))
        for j in range(n):
            exp_num_sum += row[j]
            if j != i:
                fp[i] += row[j]
                fn[j] += row[j]
            else:
                tp[i] += row[j]
    for i in range(n):
        we[i] = div(tp[i] + fp[i], exp_num_sum)
    ma_f_measure = macro_f_measure(n, tp, fp, fn, we)
    mi_f_measure = micro_f_measure(n, tp, fp, fn, we)
    print(ma_f_measure)
    print(mi_f_measure)


if __name__ == '__main__':
    solve()
