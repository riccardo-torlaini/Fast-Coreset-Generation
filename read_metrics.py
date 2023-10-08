import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as KMeans

from experiment_utils.get_data import get_dataset

DATA_NAMES = {
    'artificial': r'$c$-outlier',
    'geometric': 'Geometric',
    'benchmark': 'Benchmark',
    'blobs': 'Gaussian',
    'adult': 'Adult',
    'mnist': 'MNIST',
    'song': 'Song',
    'cover_type': 'Cover Type',
    'census': 'Census',
}
ALG_NAMES = {
    'sens_sampling': 'Sens. Sampling',
    'uniform_sampling': 'Uniform Sampling',
    'lightweight': 'Lightweight',
    'fast_coreset': 'Fast Coreset',
    'semi_uniform': 'Welterweight',
    'bico': 'BICO'
}
ALG_DICT = {
    '1': 'medians', '2': 'means'
}
COLORS = [
    'pink',
    'cyan',
    'magenta',
    'orange',
    'olive',
    'red',
    'blue',
    'yellow',
    'gray',
]

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
    if 'time' in metrics:
        # Load in the coreset construction scores
        assert 'acc' in metrics
        results_dict[method][dataset][norm][param][value]['time'] = metrics['time']
        results_dict[method][dataset][norm][param][value]['acc'] = metrics['acc']
    else:
        # Load in the coreset itself
        results_dict[method][dataset][norm][param][value]['points'] = metrics['coreset_points']
        results_dict[method][dataset][norm][param][value]['weights'] = metrics['coreset_weights']

    return results_dict, param

def read_outputs(base_path, npy_file='metrics', filter_strs=[], relevant_key=None, print_dict=False):
    if not npy_file.endswith('.npy'):
        npy_file += '.npy'

    subdirs = [d[0] for d in os.walk(base_path)]
    if filter_strs:
        filtered_dirs = subdirs
        temp_filtered = []
        for filter_str in filter_strs:
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


def make_scores_vs_param_plot(results, dataset, param, norm, value_list):
    metrics = {}
    all_methods = {}
    for method, datasets in results.items():
        # get runtimes and accuracies for k parameter on this dataset under the 1 norm
        for param_value, scores in datasets[dataset][norm][param].items():
            if method not in metrics:
                metrics[method] = {}
            metrics[method][param_value] = [0, 0]
            metrics[method][param_value][0] = np.mean(scores['acc'])
            metrics[method][param_value][1] = np.mean(scores['time'])

    loc_dict = {value: i for i, value in enumerate(value_list)}
    colors = {
        'uniform_sampling': 'red',
        'fast_coreset': 'blue',
        'sens_sampling': 'green',
        'semi_uniform': 'grey',
    }
    labels = {
        'uniform_sampling': 'Uniform Sampling',
        'fast_coreset': 'Fast-Coreset',
        'sens_sampling': 'Sensitivity Sampling',
        'semi_uniform': 'Semi-Uniform Coreset'
    }
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[method]) for method in metrics]
    labels = [labels[method] for method in metrics]
    for i, method in enumerate(metrics):
        param_values = metrics[method]
        plt.bar(
            [loc_dict[value] * len(metrics) + 1 - (len(metrics) + 1) / 2 + i for value in param_values],
            [metrics[method][value][0] for value in param_values],
            width=1,
            color=colors[method],
            edgecolor='black'
        )
    plt.legend(handles, labels)
    plt.ylabel('Coreset accuracy')
    plt.ylim([1, 1.2])
    plt.title('Effect of {} on coreset distortion'.format(param))
    plt.yticks(fontsize=10, rotation=90)
    plt.xticks(
        [i * (len(metrics)) for i in loc_dict.values()],
        list(loc_dict.keys())
    )
    plt.show()

    for i, method in enumerate(metrics):
        param_values = metrics[method]
        plt.bar(
            [loc_dict[value] * len(metrics) + 1 - (len(metrics) + 1) / 2 + i for value in param_values],
            [metrics[method][value][1] for value in param_values],
            width=1,
            color=colors[method],
            edgecolor='black'
        )
    plt.legend(handles, labels)
    plt.ylabel('Runtime in seconds')
    plt.title('Effect of {} on runtime for k-{}'.format(param, ALG_DICT[norm]))
    plt.xticks(
        [i * (len(metrics)) for i in loc_dict.values()],
        list(loc_dict.keys())
    )
    plt.show()


def get_score(points, weights, centers):
    cost = 0
    for point, weight in zip(points, weights):
        min_dist = 1000000000000
        for center in centers:
            dist = np.sum(np.square(point - center))
            if dist < min_dist:
                min_dist = dist
        cost += min_dist * weight
    return cost

def make_scores_over_datasets_plot(
    results,
    methods,
    datasets,
    param,
    norm,
    value_list,
    pattern_dict,
    y_lim,
    figure_title,
    num_centers=50
):
    metrics = {}
    for dataset in datasets:
        print('\n' + dataset)
        if dataset not in metrics:
            metrics[dataset] = {}
        full_dataset, _ = get_dataset(dataset)
        init_centers = None
        # get runtimes and accuracies for k parameter on this dataset under the 1 norm
        for method in methods:
            print(method)
            if method not in metrics[dataset]:
                metrics[dataset][method] = {}
            # If we dind't run this method on this dataset, don't try to read the results
            if dataset not in results[method]:
                continue
            for param_value, scores in results[method][dataset][norm][param].items():
                if param_value not in value_list:
                    continue
                metrics[dataset][method][param_value] = {}
                metrics[dataset][method][param_value]['means'] = [0, 0]
                metrics[dataset][method][param_value]['vars'] = [0, 0]
                if 'points' in scores:
                    # We loaded in the coresets themselves and need to run kmeans on them
                    points = scores['points']
                    weights = scores['weights']
                    if init_centers is None:
                        # If we are going to run kmeans on a coreset here, we want to initialize with the same start across all coresets
                        min_val = np.min(points)
                        max_val = np.max(points)
                        init_centers = np.random.uniform(min_val, max_val, [num_centers, int(points.shape[1])])
                    start = time()
                    kmeans_model = KMeans.KMeans(num_centers, init=init_centers, n_init=1).fit(points, sample_weight=weights)
                    cost = get_score(full_dataset, np.ones(len(full_dataset)), kmeans_model.cluster_centers_)
                    print(cost)
                    metrics[dataset][method][param_value]['means'][0] = cost
                    metrics[dataset][method][param_value]['means'][1] = time() - start
                    metrics[dataset][method][param_value]['vars'][0] = 0
                    metrics[dataset][method][param_value]['vars'][1] = 0
                else:
                    metrics[dataset][method][param_value]['means'][0] = np.mean(scores['acc'])
                    metrics[dataset][method][param_value]['means'][1] = np.mean(scores['time'])
                    metrics[dataset][method][param_value]['vars'][0] = np.var(scores['acc'])
                    metrics[dataset][method][param_value]['vars'][1] = np.var(scores['time'])
                    print(dataset, method, param_value)
                    print(metrics[dataset][method][param_value]['means'])
                    print(metrics[dataset][method][param_value]['vars'])
                    print()

    # print(metrics)

    image_dir = os.path.join('tex', 'images', norm)
    os.makedirs(image_dir, exist_ok=True)

    loc_dict = {value: i for i, value in enumerate(value_list)}
    dataset_colors = {dataset: COLORS[i] for i, dataset in enumerate(datasets)}

    # pattern_list = [ "/" , "x" , "+", "o", "O", ".", "*", "-" , "\\" , "|" ]
    pattern_list = [ " " , " " , " ", " ", " ", " ", " ", " " , " " , " " ]
    pattern_dict = {value: pattern_list[i] for i, value in enumerate(value_list)}
    pattern_handles = [plt.Rectangle((0, 0), 1, 1, hatch=pattern_dict[value]) for value in value_list]

    color_handles = [plt.Rectangle((0, 0), 1, 1, color=dataset_colors[dataset]) for dataset in datasets]
    color_labels = {DATA_NAMES[dataset]: COLORS[i] for i, dataset in enumerate(datasets)}

    plt.rcParams.update({'font.size': 16, 'text.usetex': True, 'savefig.format': 'pdf'})
    for index in [0, 1]:
        fig, axes = plt.subplots(1, len(methods))
        fig.set_figheight(5)
        fig.set_figwidth(13)
        if index == 0:
            ylabel = 'Coreset distortion'
        else:
            ylabel = 'Coreset runtime (seconds)'
        min_height = np.inf
        max_height = 0
        for i, method in enumerate(methods):
            try:
                axis = axes[i]
            except TypeError:
                # If we only have one method then there will only be one subplot
                axis = axes
            for j, dataset in enumerate(datasets):
                if method not in metrics[dataset]:
                    continue
                horizontal_shift = j * (len(value_list))
                for value in value_list:
                    try:
                        if metrics[dataset][method][value]['means'][index] < min_height and \
                                metrics[dataset][method][value]['means'][index] > 0:
                            min_height = metrics[dataset][method][value]['means'][index]
                        if metrics[dataset][method][value]['means'][index] > max_height:
                            max_height = metrics[dataset][method][value]['means'][index]
                        axis.bar(
                            loc_dict[value] + horizontal_shift,
                            metrics[dataset][method][value]['means'][index],
                            width=1,
                            color=dataset_colors[dataset],
                            edgecolor='black',
                            hatch=pattern_dict[value]
                        )
                        # axis.errorbar(
                        #     loc_dict[value] + horizontal_shift,
                        #     metrics[dataset][method][value]['means'][index],
                        #     metrics[dataset][method][value]['vars'][index],
                        #     capsize=5,
                        #     color='black',
                        # )
                    except KeyError:
                        continue
            axis.set_ylabel(ylabel)
            if index == 0:
                axis.set_ylim(y_lim)
            axis.set_yscale('log')
            axis.set_title(ALG_NAMES[method])
            axis.tick_params(
                axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False
            )
            axis.tick_params(
                axis='y',
                which='both',
                labelleft=False,
            )

        if index == 1:
            if len(methods) > 1:
                for i in range(len(methods)):
                    axes[i].set_ylim([min_height/1.5, max_height*1.5])
            else:
                axes.set_ylim([min_height/1.5, max_height*1.5])

        if len(methods) == 1:
            first_axis = axes
            last_axis = axes
        else:
            first_axis = axes[0]
            last_axis = axes[-1]
        first_axis.tick_params(
            axis='y',
            which='both',
            left=True,
            right=False,
            labelleft=True,
            labelsize=14,
            labelrotation=45
        )
        color_legend = first_axis.legend(color_handles, color_labels, loc='upper left', prop={'size': 13})
        # ax = last_axis.add_artist(color_legend)
        # pattern_legend = first_axis.legend(pattern_handles, value_list, loc='lower center')
        if index == 0:
            save_path = 'coreset_distortion-' + figure_title
        else:
            save_path = 'coreset_runtime-' + figure_title
        save_path = os.path.join(image_dir, save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)



if __name__ == '__main__':
    outputs_dir = 'outputs'
    results, params = read_outputs(
        outputs_dir,
        npy_file='metrics',
        filter_strs=['m_scalar'],
        # print_dict=True
    )

    # m_scalar_values = ['20', '40', '60', '80']
    m_scalar_values = ['40', '60', '80']
    allotted_time_values = ['0', '0.5', '1', '3', '5', '7', '10', '20']
    j_func_values = ['2', 'log', '10', 'sqrt', 'half']
    sample_method_values = ['sens', 'uniform']

    ### LOOKING AT EFFECT OF CORESET SIZE ON CORESET QUALITY for norm=2 (kmeans)
    norm = '2'
    methods = ['uniform_sampling', 'lightweight', 'semi_uniform', 'fast_coreset']
    results, params = read_outputs(
        'outputs',
        npy_file='metrics',
        filter_strs=methods,
        print_dict=True
    )
    m_scalar_pattern_dict = {
        '40': 'm=40k',
        '80': 'm = 80k',
    }
    datasets = [
        'artificial',
        'geometric',
        'benchmark',
        'blobs',
        'adult',
        'mnist',
        'song',
        'cover_type',
        'census',
    ]
    make_scores_over_datasets_plot(
        results,
        methods,
        datasets,
        'm_scalar',
        norm,
        m_scalar_values,
        m_scalar_pattern_dict,
        y_lim=[1, 10],
        figure_title='m_scalar_across_all_algorithms'
    )

    ### LOOKING AT EFFECT OF CORESET SIZE ON CORESET QUALITY for norm=2 (kmeans)
    norm = '1'
    methods = ['uniform_sampling', 'lightweight', 'semi_uniform', 'fast_coreset']
    results, params = read_outputs(
        'final_outputs',
        npy_file='metrics',
        filter_strs=methods,
        print_dict=True
    )
    m_scalar_pattern_dict = {
        '40': 'm=40k',
        '80': 'm = 80k',
    }
    datasets = [
        'artificial',
        'geometric',
        'benchmark',
        'blobs',
        'adult',
        'mnist',
        'song',
        'cover_type',
        'census',
    ]
    make_scores_over_datasets_plot(
        results,
        methods,
        datasets,
        'm_scalar',
        norm,
        m_scalar_values,
        m_scalar_pattern_dict,
        y_lim=[1, 10],
        figure_title='m_scalar_across_all_algorithms'
    )





        ### COMPARING FAST-KMEANS++ TO KMEANS++ SENSITIVITY SAMPLING
        # methods = ['sens_sampling', 'fast_coreset']
        # results, params = read_outputs(
        #     'outputs',
        #     npy_file='metrics',
        #     filter_strs=methods,
        #     print_dict=True
        # )
        # k_values = ['50', '100', '200', '400']
        # k_pattern_dict = { '10': 'k=10',
        #     '50': 'k=50',
        #     '100': 'k=100',
        #     '200': 'k=200',
        # }
        # datasets = [
        #     'geometric',
        #     'benchmark',
        #     'artificial',
        #     'blobs',
        #     'adult',
        # ]
        # make_scores_over_datasets_plot(
        #     results,
        #     methods,
        #     datasets,
        #     'k',
        #     norm,
        #     k_values,
        #     k_pattern_dict,
        #     y_lim=[1, 10000],
        #     figure_title='Effect_of_k_for_sens_sampling'
        # )





    ### Composition results
    methods = ['uniform_sampling', 'lightweight', 'semi_uniform', 'fast_coreset']
    results, params = read_outputs(
        'outputs',
        npy_file='metrics',
        filter_strs=methods,
        print_dict=True
    )
    comp_values = ['True', 'False']
    comp_pattern_dict = {'True': 'True', 'False': 'False'}
    datasets = [
        'artificial',
        'geometric',
        'benchmark',
        'blobs',
        'adult',
        'mnist',
        # 'song',
        # 'cover_type',
        # 'census',
    ]
    make_scores_over_datasets_plot(
        results,
        methods,
        datasets,
        'composition',
        '2',
        comp_values,
        comp_pattern_dict,
        y_lim=[1, 500],
        figure_title='composition'
    )





    # ### DOES USING 3 HST'S HELP?
    # methods = ['fast_coreset']
    # results, params = read_outputs(
    #     'outputs',
    #     npy_file='metrics',
    #     filter_strs=methods,
    #     print_dict=True
    # )
    # datasets = [
    #     'artificial',
    #     # 'geometric',
    #     'benchmark',
    #     # 'blobs',
    #     # 'adult',
    #     # 'mnist',
    #     # 'song',
    #     # 'cover_type',
    #     # 'census',
    # ]
    # hst_count_values = ['True', 'False']
    # hst_count_pattern_dict = {
    #     'True': 'Norm + 1 HST\'s',
    #     'False': '1 HST',
    # }
    # make_scores_over_datasets_plot(
    #     results,
    #     methods,
    #     datasets,
    #     'hst_count_from_norm',
    #     '2',
    #     hst_count_values,
    #     hst_count_pattern_dict,
    #     y_lim=[1, 1.3],
    #     figure_title='3_HSTs_vs_1_HST'
    # )




    # Run Lloyd's on coresets
    # m_scalar_pattern_dict = {
    #     '40': 'm=40k',
    #     '80': 'm = 80k',
    # }
    # datasets = [
    #     'adult',
    #     'mnist',
    #     'song',
    #     'cover_type',
    #     'census',
    # ]
    # methods = ['bico']#['uniform_sampling', 'lightweight', 'semi_uniform', 'fast_coreset', 'bico']
    # results, params = read_outputs(
    #     'outputs',
    #     npy_file='coreset',
    #     filter_strs=methods,
    #     print_dict=False
    # )
    # m_scalar_values = ['40']
    # make_scores_over_datasets_plot(
    #     results,
    #     methods,
    #     datasets,
    #     'm_scalar',
    #     '2',
    #     m_scalar_values,
    #     m_scalar_pattern_dict,
    #     y_lim=[1, 10000000000],
    #     figure_title='kmeans_costs'
    # )
