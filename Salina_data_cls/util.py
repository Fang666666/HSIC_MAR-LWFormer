import numpy as np


def print_results(n_class, oa, aa, kappa, class_acc, traintime, testtime):
    # run_times and runtime
    # output the results into a txt file and a mat file.
    n_class = n_class
    mean_oa = format(np.mean(oa * 100), '.2f')
    std_oa = format(np.std(oa * 100), '.2f')
    mean_aa = format(np.mean(aa) * 100, '.2f')
    std_aa = format(np.std(aa) * 100, '.2f')
    mean_kappa = format(np.mean(kappa) * 100, '.2f')
    std_kappa = format(np.std(kappa) * 100, '.2f')

    print('\n')
    print('train_time:', str(np.mean(traintime)), 'std:', str(np.std(traintime)))
    print('test_time:', str(np.mean(testtime)), 'std:', str(np.std(testtime)))

    for i in range(n_class):
        mean_std = str(round(np.mean(class_acc[:, i]) * 100, 2)) + '±' + str(round(np.std(class_acc[:, i]) * 100, 2))
        print('Class ', str(i + 1), ' mean ± std:', mean_std)

    print('OA mean:', str(mean_oa), 'std:', str(std_oa))
    print('AA mean:', str(mean_aa), 'std:', str(std_aa))
    print('Kappa mean:', str(mean_kappa), 'std:', str(std_kappa))
