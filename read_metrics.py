import os
import numpy as np
import matplotlib.pyplot as plt

ALG_DICT = {
    '1': 'medians', '2': 'means'
}
COLORS = [
    'red',
    'blue',
    'yellow',
    'green',
    'pink',
    'cyan',
    'magenta',
    'orange',
    'olive'
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
    results_dict[method][dataset][norm][param][value]['time'] = metrics['time']
    results_dict[method][dataset][norm][param][value]['acc'] = metrics['acc']

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
            metrics[method][param_value][0] = scores['acc']
            metrics[method][param_value][1] = scores['time']

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



def make_scores_over_datasets_plot(
    results,
    methods,
    datasets,
    param,
    norm,
    value_list,
    pattern_dict,
    y_lim,
    figure_title
):
    metrics = {}
    for dataset in datasets:
        if dataset not in metrics:
            metrics[dataset] = {}
        # get runtimes and accuracies for k parameter on this dataset under the 1 norm
        for method in methods:
            if method not in metrics[dataset]:
                metrics[dataset][method] = {}
            # If we dind't run this method on this dataset, don't try to read the results
            if dataset not in results[method]:
                continue
            for param_value, scores in results[method][dataset][norm][param].items():
                metrics[dataset][method][param_value] = [0, 0]
                metrics[dataset][method][param_value][0] = scores['acc']
                metrics[dataset][method][param_value][1] = scores['time']

    image_dir = os.path.join('tex', 'images')
    os.makedirs(image_dir, exist_ok=True)

    loc_dict = {value: i for i, value in enumerate(value_list)}
    dataset_colors = {dataset: COLORS[i] for i, dataset in enumerate(datasets)}

    pattern_list = [ "/" , "x" , "+", "o", "O", ".", "*", "-" , "\\" , "|" ]
    pattern_dict = {value: pattern_list[i] for i, value in enumerate(value_list)}
    pattern_handles = [plt.Rectangle((0, 0), 1, 1, hatch=pattern_dict[value]) for value in value_list]

    color_handles = [plt.Rectangle((0, 0), 1, 1, color=dataset_colors[dataset]) for dataset in datasets]
    color_labels = {dataset: COLORS[i] for i, dataset in enumerate(datasets)}

    plt.rcParams.update({'font.size': 10, 'text.usetex': True, 'savefig.format': 'pdf'})
    for index in [0, 1]:
        fig, axes = plt.subplots(1, len(methods))
        fig.set_figheight(5.5)
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
                        if metrics[dataset][method][value][index] < min_height and \
                                metrics[dataset][method][value][index] > 0:
                            min_height = metrics[dataset][method][value][index]
                        if metrics[dataset][method][value][index] > max_height:
                            max_height = metrics[dataset][method][value][index]
                        axis.bar(
                            loc_dict[value] + horizontal_shift,
                            metrics[dataset][method][value][index],
                            width=1,
                            color=dataset_colors[dataset],
                            edgecolor='black',
                            hatch=pattern_dict[value]
                        )
                    except KeyError:
                        continue
            axis.set_ylabel(ylabel)
            if index == 0:
                axis.set_ylim(y_lim)
            axis.set_yscale('log')
            axis.set_title(method)
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
                labelleft=False
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
            labelleft=True
        )
        color_legend = last_axis.legend(color_handles, color_labels, loc='upper right')
        ax = last_axis.add_artist(color_legend)
        pattern_legend = first_axis.legend(pattern_handles, value_list, loc='upper left')
        if index == 0:
            save_path = 'coreset_distortion-' + figure_title
        else:
            save_path = 'coreset_runtime-' + figure_title
        save_path = os.path.join('tex', 'images', save_path)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)



if __name__ == '__main__':
    # Comment out to read optimization_times for uniform_umap
    outputs_dir = 'outputs'
    results, params = read_outputs(
        outputs_dir,
        npy_file='metrics',
        filter_strs=['semi_uniform'],
        print_dict=True
    )

    m_scalar_values = ['20', '40', '60', '80']
    # m_scalar_values = ['40', '60', '80']
    allotted_time_values = ['0', '0.5', '1', '3', '5', '7', '10', '20']
    j_func_values = ['2', 'log', '10', 'sqrt', 'half']
    sample_method_values = ['sens', 'uniform']

    norm = '2'
    # List of datasets --- ['blobs', 'benchmark', 'mnist', 'artificial', 'census']

    dataset = 'census'
    # make_scores_vs_param_plot(results, dataset, 'j_func', norm, j_func_values)
    # make_scores_vs_param_plot(results, dataset, 'sample_method', norm, sample_method_values)


    ### LOOKING AT EFFECT OF CORESET SIZE ON CORESET QUALITY
    norm = '2'
    methods = ['semi_uniform', 'fast_coreset', 'uniform_sampling', 'lightweight']
    results, params = read_outputs(
        outputs_dir,
        npy_file='metrics',
        filter_strs=methods,
        print_dict=True
    )
    m_scalar_pattern_dict = {
        '5': 'm=5k',
        '10': 'm=10k',
        '20': 'm=20k',
        '40': 'm=40k',
        '60': 'm=60k',
        '80': 'm = 80k',
    }
    datasets = [
        'artificial',
        'geometric',
        'benchmark',
        'blobs',
        'mnist',
        'census',
        # 'kdd_cup',
        'song',
        'cover_type'
    ]
    # Coreset size plots
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
    methods = ['fast_coreset', 'sens_sampling']
    results, params = read_outputs(
        outputs_dir,
        npy_file='metrics',
        filter_strs=methods,
        print_dict=True
    )
    k_values = ['10', '50', '100', '200']
    k_pattern_dict = {
        '10': 'k=10',
        '50': 'k=50',
        '100': 'k=100',
        '200': 'k=200',
    }
    datasets = [
        'artificial',
        'geometric',
        'benchmark',
        'blobs',
        'mnist',
        'census',
    ]
    m_scalar_values = ['40', '60', '80']
    make_scores_over_datasets_plot(
        results,
        methods,
        datasets,
        'm_scalar',
        norm,
        m_scalar_values,
        m_scalar_pattern_dict,
        y_lim=[1, 1.3],
        figure_title='m_scalar_for_sens_sampling'
    )
    make_scores_over_datasets_plot(
        results,
        methods,
        datasets,
        'k',
        norm,
        k_values,
        k_pattern_dict,
        y_lim=[1, 1.3],
        figure_title='Effect_of_k_for_sens_sampling'
    )



    ### DOES USING 3 HST'S HELP?
    methods = ['fast_coreset']
    results, params = read_outputs(
        outputs_dir,
        npy_file='metrics',
        filter_strs=methods,
        print_dict=True
    )
    datasets = [
        'artificial',
        'geometric',
        'benchmark',
        'blobs',
        'mnist',
        'census',
        'song',
        'cover_type'
    ]
    hst_count_values = ['True', 'False']
    hst_count_pattern_dict = {
        'True': 'Norm + 1 HST\'s',
        'False': '1 HST',
    }
    make_scores_over_datasets_plot(
        results,
        methods,
        datasets,
        'hst_count_from_norm',
        norm,
        hst_count_values,
        hst_count_pattern_dict,
        y_lim=[1, 1.3],
        figure_title='3_HSTs_vs_1_HST'
    )

