import os
import numpy as np
import matplotlib.pyplot as plt

ALG_DICT = {
    '1': 'medians',
    '2': 'means'
}

def update_results_dict(results_dict, directory, metrics):
    path = directory.split('/')
    method = path[1]
    dataset = path[2]
    norm = path[3]
    param = path[4]
    value = path[5]
    if method not in results_dict:
        results_dict[method] = {}
    if dataset not in results_dict[method]:
        results_dict[method][dataset] = {}
    if norm not in results_dict[method][dataset]:
        results_dict[method][dataset][norm] = {}
    if param not in results_dict[method][dataset][norm]:
        results_dict[method][dataset][norm][param] = {}
    if value not in results_dict[method][dataset][norm][param]:
        results_dict[method][dataset][norm][param][value] = {}
    results_dict[method][dataset][norm][param][value]['time'] = metrics['time']
    results_dict[method][dataset][norm][param][value]['acc'] = metrics['acc']

    return results_dict, param

def read_outputs(base_path, npy_file='metrics', filter_strs=[], relevant_key=None, print_dict=False):
    if not npy_file.endswith('.npy'):
        npy_file += '.npy'

    subdirs = [d[0] for d in os.walk(base_path)]
    if filter_strs:
        filtered_dirs = subdirs
        for filter_str in filter_strs:
            temp_filtered = []
            for sd in filtered_dirs:
                if filter_str in sd:
                    temp_filtered += [sd]
            filtered_dirs = temp_filtered
    else:
        filtered_dirs = subdirs

    if not filtered_dirs:
        raise ValueError("No output directories available after filtering")

    all_results = {}
    all_params = []
    for directory in filtered_dirs:
        if not os.path.isfile(os.path.join(directory, npy_file)):
            continue
        metrics = np.load(os.path.join(directory, npy_file), allow_pickle=True)
        metrics = metrics[()]
        all_results, param = update_results_dict(all_results, directory, metrics)
        if param not in all_params:
            all_params += [param]
        if print_dict:
            if relevant_key is None:
                print(directory)
                for metric, val in metrics.items():
                    print('%s:' % metric, val)
                print()
            else:
                try:
                    print('%s --- %s =' % (directory, relevant_key), metrics[relevant_key])
                except KeyError:
                    print('WARNING --- key %s not present in %s' % (relevant_key, os.path.join(directory, npy_file)))

    return all_results, all_params


def make_scores_vs_k_plot(results, dataset, norm='2'):
    metrics = {}
    all_methods = {}
    for method, datasets in results.items():
        # get runtimes and accuracies for k parameter on this dataset under the 1 norm
        for param_value, scores in datasets[dataset][norm]['k'].items():
            if method not in metrics:
                metrics[method] = {}
            metrics[method][param_value] = [0, 0]
            metrics[method][param_value][0] = scores['acc']
            metrics[method][param_value][1] = scores['time']

    k_loc_dict = {
        '10': 1,
        '40': 2,
        '100': 3,
    }
    colors = {
        'uniform_sampling': 'red',
        'fast_coreset': 'blue',
        'sens_sampling': 'green'
    }
    labels = {
        'uniform_sampling': 'Uniform Sampling',
        'fast_coreset': 'Fast Coreset',
        'sens_sampling': 'Sensitivity Sampling'
    }
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method]) for method in metrics]
    labels = [labels[method] for method in metrics]
    for i, method in enumerate(metrics):
        param_values = metrics[method]
        plt.bar(
            [k_loc_dict[value] * len(metrics) + 1 - (len(metrics) + 1) / 2 + i for value in param_values],
            [metrics[method][value][0] for value in param_values],
            width=1.0,
            color=colors[method]
        )
    plt.legend(handles, labels)
    plt.ylabel('Coreset accuracy')
    plt.title('Effect of k on coreset accuracy for k-{}'.format(ALG_DICT[norm]))
    plt.ylim([1.0, 1.075])
    plt.xticks(
        [i * (len(metrics)) for i in k_loc_dict.values()],
        list(k_loc_dict.keys())
    )
    plt.show()

    for i, method in enumerate(metrics):
        param_values = metrics[method]
        plt.bar(
            [k_loc_dict[value] * len(metrics) + 1 - (len(metrics) + 1) / 2 + i for value in param_values],
            [metrics[method][value][1] for value in param_values],
            width=1,
            color=colors[method]
        )
    plt.legend(handles, labels)
    plt.ylabel('Runtime in seconds')
    plt.xticks(
        [i * (len(metrics)) for i in k_loc_dict.values()],
        list(k_loc_dict.keys())
    )
    plt.title('Effect of k on runtime for k-{}'.format(ALG_DICT[norm]))
    plt.show()


def make_scores_vs_eps_plot(results, dataset, norm='2'):
    metrics = {}
    all_methods = {}
    for method, datasets in results.items():
        # get runtimes and accuracies for k parameter on this dataset under the 1 norm
        for param_value, scores in datasets[dataset][norm]['eps'].items():
            if method not in metrics:
                metrics[method] = {}
            metrics[method][param_value] = [0, 0]
            metrics[method][param_value][0] = scores['acc']
            metrics[method][param_value][1] = scores['time']

    eps_loc_dict = {
        '0.2': 1,
        '0.4': 2,
        '0.8': 3,
    }
    colors = {
        'uniform_sampling': 'red',
        'fast_coreset': 'blue',
        'sens_sampling': 'green'
    }
    labels = {
        'uniform_sampling': 'Uniform Sampling',
        'fast_coreset': 'Sensitivity Sampling',
        'sens_sampling': 'Fast-Coreset'
    }
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method]) for method in metrics]
    labels = [labels[method] for method in metrics]
    for i, method in enumerate(metrics):
        param_values = metrics[method]
        plt.bar(
            [eps_loc_dict[value] * len(metrics) + 1 - (len(metrics) + 1) / 2 + i for value in param_values],
            [metrics[method][value][0] for value in param_values],
            width=1,
            color=colors[method]
        )
    plt.legend(handles, labels)
    plt.ylabel('Coreset accuracy')
    plt.title('Effect of epsilon on coreset accuracy')
    plt.ylim([1.0, 1.07])
    plt.xticks(
        [i * (len(metrics)) for i in eps_loc_dict.values()],
        list(eps_loc_dict.keys())
    )
    plt.show()

    for i, method in enumerate(metrics):
        param_values = metrics[method]
        plt.bar(
            [eps_loc_dict[value] * len(metrics) + 1 - (len(metrics) + 1) / 2 + i for value in param_values],
            [metrics[method][value][1] for value in param_values],
            width=1,
            color=colors[method]
        )
    plt.legend(handles, labels)
    plt.ylabel('Runtime in seconds')
    plt.title('Effect of epsilon on runtime for k-{}'.format(ALG_DICT[norm]))
    plt.xticks(
        [i * (len(metrics)) for i in eps_loc_dict.values()],
        list(eps_loc_dict.keys())
    )
    plt.show()


def make_scores_vs_oversample_plot(results, dataset, norm='2'):
    metrics = {}
    all_methods = {}
    for method, datasets in results.items():
        # get runtimes and accuracies for k parameter on this dataset under the 1 norm
        for param_value, scores in datasets[dataset][norm]['oversample'].items():
            if method not in metrics:
                metrics[method] = {}
            metrics[method][param_value] = [0, 0]
            metrics[method][param_value][0] = scores['acc']
            metrics[method][param_value][1] = scores['time']

    eps_loc_dict = {
        '1': 1,
        '2': 2,
        '4': 3,
        '8': 4,
    }
    colors = {
        'uniform_sampling': 'red',
        'fast_coreset': 'blue',
        'sens_sampling': 'green'
    }
    labels = {
        'uniform_sampling': 'Uniform Sampling',
        'fast_coreset': 'Fast-Coreset',
        'sens_sampling': 'Sensitivity Sampling'
    }
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method]) for method in metrics]
    labels = [labels[method] for method in metrics]
    for i, method in enumerate(metrics):
        param_values = metrics[method]
        plt.bar(
            [eps_loc_dict[value] * len(metrics) + 1 - (len(metrics) + 1) / 2 + i for value in param_values],
            [metrics[method][value][0] for value in param_values],
            width=1,
            color=colors[method]
        )
    plt.legend(handles, labels)
    plt.ylabel('Coreset accuracy')
    plt.title('Effect of Oversample scalar on coreset accuracy')
    plt.xticks(
        [i * (len(metrics)) for i in eps_loc_dict.values()],
        list(eps_loc_dict.keys())
    )
    plt.show()

    for i, method in enumerate(metrics):
        param_values = metrics[method]
        plt.bar(
            [eps_loc_dict[value] * len(metrics) + 1 - (len(metrics) + 1) / 2 + i for value in param_values],
            [metrics[method][value][1] for value in param_values],
            width=1,
            color=colors[method]
        )
    plt.legend(handles, labels)
    plt.ylabel('Runtime in seconds')
    plt.title('Effect of epsilon on runtime for k-{}'.format(ALG_DICT[norm]))
    plt.xticks(
        [i * (len(metrics)) for i in eps_loc_dict.values()],
        list(eps_loc_dict.keys())
    )
    plt.show()




if __name__ == '__main__':
    # Comment out to read optimization_times for uniform_umap
    results, params = read_outputs(
        'outputs',
        npy_file='metrics',
        filter_strs=[''],
    )

    norm = '2'
    dataset = 'blobs'
    # make_scores_vs_k_plot(results, dataset, norm)
    # make_scores_vs_oversample_plot(results, dataset, norm)
    make_scores_vs_eps_plot(results, dataset, norm)
