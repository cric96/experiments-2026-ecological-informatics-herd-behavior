import os
import numpy as np
import xarray as xr
import re
from pathlib import Path
import collections


def distance(val, ref):
    return abs(ref - val)


vectDistance = np.vectorize(distance)


def cmap_xmap(function, cmap):
    """ Applies function, on the indices of colormap cmap. Beware, function
    should map the [0, 1] segment to itself, or you are in for surprises.

    See also cmap_xmap.
    """
    cdict = cmap._segmentdata
    function_to_map = lambda x: (function(x[0]), x[1], x[2])
    for key in ('red', 'green', 'blue'):
        cdict[key] = map(function_to_map, cdict[key])
    #        cdict[key].sort()
    #        assert (cdict[key][0]<0 or cdict[key][-1]>1), "Resulting indices extend out of the [0, 1] segment."
    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)


def getClosest(sortedMatrix, column, val):
    while len(sortedMatrix) > 3:
        half = int(len(sortedMatrix) / 2)
        sortedMatrix = sortedMatrix[-half - 1:] if sortedMatrix[half, column] < val else sortedMatrix[: half + 1]
    if len(sortedMatrix) == 1:
        result = sortedMatrix[0].copy()
        result[column] = val
        return result
    else:
        safecopy = sortedMatrix.copy()
        safecopy[:, column] = vectDistance(safecopy[:, column], val)
        minidx = np.argmin(safecopy[:, column])
        safecopy = safecopy[minidx, :].A1
        safecopy[column] = val
        return safecopy


def convert(column, samples, matrix):
    return np.matrix([getClosest(matrix, column, t) for t in samples])


def valueOrEmptySet(k, d):
    return (d[k] if isinstance(d[k], set) else {d[k]}) if k in d else set()


def mergeDicts(d1, d2):
    """
    Creates a new dictionary whose keys are the union of the keys of two
    dictionaries, and whose values are the union of values.

    Parameters
    ----------
    d1: dict
        dictionary whose values are sets
    d2: dict
        dictionary whose values are sets

    Returns
    -------
    dict
        A dict whose keys are the union of the keys of two dictionaries,
    and whose values are the union of values

    """
    res = {}
    for k in d1.keys() | d2.keys():
        res[k] = valueOrEmptySet(k, d1) | valueOrEmptySet(k, d2)
    return res


def extractCoordinates(filename):
    """
    Scans the header of an Alchemist file in search of the variables.

    Parameters
    ----------
    filename : str
        path to the target file
    mergewith : dict
        a dictionary whose dimensions will be merged with the returned one

    Returns
    -------
    dict
        A dictionary whose keys are strings (coordinate name) and values are
        lists (set of variable values)

    """
    with open(filename, 'r') as file:
        #        regex = re.compile(' (?P<varName>[a-zA-Z._-]+) = (?P<varValue>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),?')
        regex = r"(?P<varName>[a-zA-Z._-]+) = (?P<varValue>[^,]*),?"
        dataBegin = r"\d"
        is_float = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        for line in file:
            match = re.findall(regex, line.replace('Infinity', '1e30000'))
            if match:
                return {
                    var: float(value) if re.match(is_float, value)
                    else bool(re.match(r".*?true.*?", value.lower())) if re.match(r".*?(true|false).*?", value.lower())
                    else value
                    for var, value in match
                }
            elif re.match(dataBegin, line[0]):
                return {}


def extractVariableNames(filename):
    """
    Gets the variable names from the Alchemist data files header.

    Parameters
    ----------
    filename : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    with open(filename, 'r') as file:
        dataBegin = re.compile(r'\d')
        lastHeaderLine = ''
        for line in file:
            if dataBegin.match(line[0]):
                break
            else:
                lastHeaderLine = line
        if lastHeaderLine:
            regex = re.compile(r' (?P<varName>\S+)')
            return regex.findall(lastHeaderLine)
        return []


def openCsv(path):
    """
    Converts an Alchemist export file into a list of lists representing the matrix of values.

    Parameters
    ----------
    path : str
        path to the target file

    Returns
    -------
    list of list
        A matrix with the values of the csv file

    """
    regex = re.compile(r'\d')
    with open(path, 'r') as file:
        lines = filter(lambda x: regex.match(x[0]), file.readlines())
        return [[float(x) for x in line.split()] for line in lines]


def beautifyValue(v):
    """
    Converts an object to a better version for printing, in particular:
        - if the object converts to float, then its float value is used
        - if the object can be rounded to int, then the int value is preferred

    Parameters
    ----------
    v : object
        the object to try to beautify

    Returns
    -------
    object or float or int
        the beautified value
    """
    try:
        v = float(v)
        if v.is_integer():
            return int(v)
        return v
    except:
        return v


if __name__ == '__main__':
    # CONFIGURE SCRIPT
    # Where to find Alchemist data files (override with DATA_DIR for a different run,
    # e.g. the realistic variant that exports to data-realistic/).
    directory = os.environ.get('DATA_DIR', 'data')
    # Where to save charts (override with OUT_DIR to keep runs side by side).
    output_directory = os.environ.get('OUT_DIR', 'charts')
    # How to name the summary of the processed data (override with PICKLE_PREFIX so a
    # second run does not clobber the validated data_summary_* pickle).
    pickleOutput = os.environ.get('PICKLE_PREFIX', 'data_summary')
    # Experiment prefixes: one per experiment (root of the file name)
    experiments = ['velocity_simulation']
    floatPrecision = '{: 0.3f}'
    # Number of time samples
    timeSamples = 10
    # time management
    minTime = 0
    maxTime = 1800
    timeColumnName = 'time'
    logarithmicTime = False
    # One or more variables are considered random and "flattened"
    seedVars = ['seed']  # ['seed']


    # Label mapping
    class Measure:
        def __init__(self, description, unit=None):
            self.__description = description
            self.__unit = unit

        def description(self):
            return self.__description

        def unit(self):
            return '' if self.__unit is None else f'({self.__unit})'

        def derivative(self, new_description=None, new_unit=None):
            def cleanMathMode(s):
                return s[1:-1] if s[0] == '$' and s[-1] == '$' else s

            def deriveString(s):
                return r'$d ' + cleanMathMode(s) + r'/{dt}$'

            def deriveUnit(s):
                return f'${cleanMathMode(s)}' + '/{s}$' if s else None

            result = Measure(
                new_description if new_description else deriveString(self.__description),
                new_unit if new_unit else deriveUnit(self.__unit),
            )
            return result

        def __str__(self):
            return f'{self.description()} {self.unit()}'


    centrality_label = 'H_a(x)'


    def expected(x):
        return r'\mathbf{E}[' + x + ']'


    def stdev_of(x):
        return r'\sigma{}[' + x + ']'


    def mse(x):
        return 'MSE[' + x + ']'


    def cardinality(x):
        return r'\|' + x + r'\|'


    labels = {
        'nodeCount': Measure(r'$n$', 'nodes'),
        'harmonicCentrality[Mean]': Measure(f'${expected("H(x)")}$'),
        'meanNeighbors': Measure(f'${expected(cardinality("N"))}$', 'nodes'),
        'speed': Measure(r'$\|\vec{v}\|$', r'$m/s$'),
        'msqer@harmonicCentrality[Max]': Measure(r'$\max{(' + mse(centrality_label) + ')}$'),
        'msqer@harmonicCentrality[Min]': Measure(r'$\min{(' + mse(centrality_label) + ')}$'),
        'msqer@harmonicCentrality[Mean]': Measure(f'${expected(mse(centrality_label))}$'),
        'msqer@harmonicCentrality[StandardDeviation]': Measure(f'${stdev_of(mse(centrality_label))}$'),
        'org:protelis:tutorial:distanceTo[max]': Measure(r'$m$', 'max distance'),
        'org:protelis:tutorial:distanceTo[mean]': Measure(r'$m$', 'mean distance'),
        'org:protelis:tutorial:distanceTo[min]': Measure(r'$m$', ',min distance'),
    }


    def derivativeOrMeasure(variable_name):
        if variable_name.endswith('dt'):
            return labels.get(variable_name[:-2], Measure(variable_name)).derivative()
        return Measure(variable_name)


    def label_for(variable_name):
        return labels.get(variable_name, derivativeOrMeasure(variable_name)).description()


    def unit_for(variable_name):
        return str(labels.get(variable_name, derivativeOrMeasure(variable_name)))


    # Setup libraries
    np.set_printoptions(formatter={'float': floatPrecision.format})
    # Read the last time the data was processed, reprocess only if new data exists, otherwise just load
    import pickle
    import os

    timeProcessedFile = f'{pickleOutput}_timeprocessed'
    if os.path.exists(directory):
        newestFileTime = max([os.path.getmtime(directory + '/' + file) for file in os.listdir(directory)], default=0.0)
        try:
            lastTimeProcessed = pickle.load(open(timeProcessedFile, 'rb'))
        except:
            lastTimeProcessed = -1
        shouldRecompute = not os.path.exists(".skip_data_process") and newestFileTime != lastTimeProcessed
        if not shouldRecompute:
            try:
                means = pickle.load(open(pickleOutput + '_mean', 'rb'))
                stdevs = pickle.load(open(pickleOutput + '_std', 'rb'))
            except:
                shouldRecompute = True
        if shouldRecompute:
            timefun = np.logspace if logarithmicTime else np.linspace
            means = {}
            stdevs = {}
            for experiment in experiments:
                # Collect all files for the experiment of interest
                import fnmatch

                allfiles = filter(lambda file: fnmatch.fnmatch(file, experiment + '_*.csv'), os.listdir(directory))
                allfiles = [directory + '/' + name for name in allfiles]
                allfiles.sort()
                # From the file name, extract the independent variables
                dimensions = {}
                for file in allfiles:
                    dimensions = mergeDicts(dimensions, extractCoordinates(file))
                dimensions = {k: sorted(v) for k, v in dimensions.items()}
                # Add time to the independent variables
                dimensions[timeColumnName] = range(0, timeSamples)
                # Compute the matrix shape
                shape = tuple(len(v) for k, v in dimensions.items())
                # Prepare the Dataset
                dataset = xr.Dataset()
                for k, v in dimensions.items():
                    dataset.coords[k] = v
                if len(allfiles) == 0:
                    print("WARNING: No data for experiment " + experiment)
                    means[experiment] = dataset
                    stdevs[experiment] = xr.Dataset()
                else:
                    varNames = extractVariableNames(allfiles[0])
                    for v in varNames:
                        if v != timeColumnName:
                            novals = np.ndarray(shape)
                            novals.fill(float('nan'))
                            dataset[v] = (dimensions.keys(), novals)
                    # Compute maximum and minimum time, create the resample
                    timeColumn = varNames.index(timeColumnName)
                    allData = {file: np.matrix(openCsv(file)) for file in allfiles}
                    computeMin = minTime is None
                    computeMax = maxTime is None
                    if computeMax:
                        maxTime = float('-inf')
                        for data in allData.values():
                            maxTime = max(maxTime, data[-1, timeColumn])
                    if computeMin:
                        minTime = float('inf')
                        for data in allData.values():
                            minTime = min(minTime, data[0, timeColumn])
                    timeline = timefun(minTime, maxTime, timeSamples)
                    # Resample
                    for file in allData:
                        #                    print(file)
                        allData[file] = convert(timeColumn, timeline, allData[file])
                    # Populate the dataset
                    for file, data in allData.items():
                        dataset[timeColumnName] = timeline
                        for idx, v in enumerate(varNames):
                            if v != timeColumnName:
                                darray = dataset[v]
                                experimentVars = extractCoordinates(file)
                                darray.loc[experimentVars] = data[:, idx].A1
                    # Fold the dataset along the seed variables, producing the mean and stdev datasets
                    mergingVariables = [seed for seed in seedVars if seed in dataset.coords]
                    means[experiment] = dataset.mean(dim=mergingVariables, skipna=True)
                    stdevs[experiment] = dataset.std(dim=mergingVariables, skipna=True)
            # Save the datasets
            pickle.dump(means, open(pickleOutput + '_mean', 'wb'), protocol=-1)
            pickle.dump(stdevs, open(pickleOutput + '_std', 'wb'), protocol=-1)
            pickle.dump(newestFileTime, open(timeProcessedFile, 'wb'))
    else:
        means = {experiment: xr.Dataset() for experiment in experiments}
        stdevs = {experiment: xr.Dataset() for experiment in experiments}

    # QUICK CHARTING

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx

    matplotlib.rcParams.update({'axes.titlesize': 12})
    matplotlib.rcParams.update({'axes.labelsize': 10})


    def make_line_chart(
            xdata,
            ydata,
            title=None,
            ylabel=None,
            xlabel=None,
            colors=None,
            linewidth=1,
            error_alpha=0.2,
            figure_size=(6, 4)
    ):
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #        ax.set_ylim(0)
        #        ax.set_xlim(min(xdata), max(xdata))
        index = 0
        for (label, (data, error)) in ydata.items():
            #            print(f'plotting {data}\nagainst {xdata}')
            lines = ax.plot(xdata, data, label=label, color=colors(index / (len(ydata) - 1)) if colors else None,
                            linewidth=linewidth)
            index += 1
            if error is not None:
                last_color = lines[-1].get_color()
                ax.fill_between(
                    xdata,
                    data + error,
                    data - error,
                    facecolor=last_color,
                    alpha=error_alpha,
                )
        return (fig, ax)


    def generate_all_charts(means, errors=None, basedir=''):
        viable_coords = {coord for coord in means.coords if means[coord].size > 1}
        for comparison_variable in viable_coords - {timeColumnName}:
            mergeable_variables = viable_coords - {timeColumnName, comparison_variable}
            for current_coordinate in mergeable_variables:
                merge_variables = mergeable_variables - {current_coordinate}
                merge_data_view = means.mean(dim=merge_variables, skipna=True)
                merge_error_view = errors.mean(dim=merge_variables, skipna=True)
                for current_coordinate_value in merge_data_view[current_coordinate].values:
                    beautified_value = beautifyValue(current_coordinate_value)
                    for current_metric in merge_data_view.data_vars:
                        title = f'{label_for(current_metric)} for diverse {label_for(comparison_variable)} when {label_for(current_coordinate)}={beautified_value}'
                        for withErrors in [True, False]:
                            fig, ax = make_line_chart(
                                title=title,
                                xdata=merge_data_view[timeColumnName],
                                xlabel=unit_for(timeColumnName),
                                ylabel=unit_for(current_metric),
                                ydata={
                                    beautifyValue(label): (
                                        merge_data_view.sel(selector)[current_metric],
                                        merge_error_view.sel(selector)[current_metric] if withErrors else 0
                                    )
                                    for label in merge_data_view[comparison_variable].values
                                    for selector in
                                    [{comparison_variable: label, current_coordinate: current_coordinate_value}]
                                },
                            )
                            ax.set_xlim(minTime, maxTime)
                            ax.legend()
                            fig.tight_layout()
                            by_time_output_directory = f'{output_directory}/{basedir}/{comparison_variable}'
                            Path(by_time_output_directory).mkdir(parents=True, exist_ok=True)
                            figname = f'{comparison_variable}_{current_metric}_{current_coordinate}_{beautified_value}{"_err" if withErrors else ""}'
                            for symbol in r".[]\/@:":
                                figname = figname.replace(symbol, '_')
                            fig.savefig(f'{by_time_output_directory}/{figname}.pdf')
                            plt.close(fig)


    for experiment in experiments:
        current_experiment_means = means[experiment]
        current_experiment_errors = stdevs[experiment]
        #generate_all_charts(current_experiment_means, current_experiment_errors, basedir = f'{experiment}/all')

    # Custom charting

    import seaborn as sns
    import pandas as pd
    import seaborn.objects as so
    from scipy import stats
    from seaborn import axes_style

    os.makedirs(output_directory, exist_ok=True)

    so.Plot.config.theme.update(axes_style("whitegrid"))

    sns.color_palette("viridis", as_cmap=True)
    # Shared paper style — identical to compare_distributions.py so every figure across
    # both scripts looks like it belongs to the same paper (same theme, fonts, weights).
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType (journal-safe), avoid Type-3
        "savefig.bbox": "tight",
        "axes.titlesize": "medium",
        "legend.frameon": True,
    })

    # --- Shared comparison settings -------------------------------------------------
    # Both datasets are treated identically so the simulated and real plots can be read
    # on the same axes: same unit (m/s), same velocity floor, same smoothing, same range.
    FLOOR_KMH = 1.0                 # the KABR tracker cannot resolve motion below 1 km/h
    FLOOR_MS = FLOOR_KMH / 3.6      # ~0.278 m/s
    BW = 1.8                        # KDE smoothing, identical for sim and real (same as curve_similarity)
    XMAX_MS = 4.0                   # covers the simulated support (~3.8 m/s) and all real data
    CLIP = (FLOOR_MS, XMAX_MS)

    # --- Simulated data -------------------------------------------------------------
    dataset = means['velocity_simulation'].to_dataframe().reset_index()
    dataset = dataset.melt(id_vars=['time', 'intrinsicForwardCoefficient', 'intrinsicLateralMultiplier', 'NumberOfHerds'],
                                  value_vars=[col for col in dataset.columns if col.startswith('node-')],
                                  var_name='node', value_name='velocity').dropna()
    # Drop the t=0 initialisation artefact (every agent starts at velocity 0).
    dataset = dataset[dataset['time'] > 0]
    dataset['velocity'] = dataset['velocity'] / 3.6  # convert to m/s
    # Remove the "zero"/stationary spike so the simulated support matches the real one,
    # which is censored at 1 km/h. Without this the sim carries a large mass of agents
    # below the tracker resolution that the real data can never contain.
    dataset = dataset[dataset['velocity'] >= FLOOR_MS]

    n = 2  # Define the interval for removal
    all_coefficients = sorted(dataset['intrinsicForwardCoefficient'].unique())
    coefficients_to_keep = [coef for i, coef in enumerate(all_coefficients) if i % n == 0]
    dataset = dataset[dataset['intrinsicForwardCoefficient'].isin(coefficients_to_keep)]

    plt.figure(figsize=(10, 6), layout='tight')
    g = sns.FacetGrid(
        dataset,
        col='intrinsicLateralMultiplier',
        aspect=1.7,
        height=5,
        col_wrap=3,
        sharex=True,
        sharey=True,
    )
    # common_norm=True keeps the RELATIVE mass of each forward coefficient instead of
    # rescaling every curve to unit area; this way tall/short curves reflect how much
    # probability each parameter actually puts at a given speed and can be read against
    # the real distribution on the same y-scale.
    g.map_dataframe(
        sns.kdeplot,
        x='velocity',
        hue='intrinsicForwardCoefficient',
        linewidth=0.8,
        palette='viridis',
        bw_adjust=BW,
        clip=CLIP,
        common_norm=True,
    )
    for ax in g.axes.flat:
        ax.set_xlim(0, XMAX_MS)
    g.set_titles("LateralVelocityMultiplier={col_name}")
    g.set_xlabels("Velocity (m/s)")
    # Colour legend: intrinsicForwardCoefficient is a continuous swept variable, so the
    # colours are a viridis ramp — a colorbar is the correct legend for them.
    norm = matplotlib.colors.Normalize(vmin=min(coefficients_to_keep),
                                       vmax=max(coefficients_to_keep))
    sm = cmx.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    g.figure.subplots_adjust(right=0.9)
    g.figure.colorbar(sm, ax=list(g.axes.flat), label='intrinsicForwardCoefficient')
    g.savefig(f"{output_directory}/velocity_simulation.pdf")

    # --- Real velocity plot ---------------------------------------------------------
    real_dataset = pd.read_csv("data/velocity_reconstruction.csv", delimiter=' ', comment='#', names=list(range(10)))
    real_dataset.rename(columns={0: 'time'}, inplace=True)
    real_dataset = real_dataset.melt(id_vars=['time'], var_name='node', value_name='velocity').dropna()
    real_dataset['velocity'] = real_dataset['velocity'] / 3.6  # convert to m/s
    # "Togliere lo zero": drop the 1 km/h floor spike (~20% of samples), i.e. the
    # animals sitting at the tracker's minimum resolvable speed. What remains is the
    # distribution of genuinely moving animals, directly comparable to the sim above.
    real_dataset = real_dataset[real_dataset['velocity'] > FLOOR_MS]

    plt.figure(figsize=(10, 6), layout='tight')
    real_plot = sns.kdeplot(data=real_dataset, x='velocity', color='#2a78d6',
                            bw_adjust=BW, clip=CLIP)
    plt.xlim(0, XMAX_MS)
    plt.xlabel("Velocity (m/s)")
    real_plot.get_figure().savefig(f"{output_directory}/velocity_reconstruction.pdf")

    # --- Direct overlay: real vs pooled simulation ----------------------------------
    # A single figure so the two can be compared at a glance on identical axes. Each
    # curve is a probability density (area 1) because the sample sizes differ by orders
    # of magnitude; the shapes, modes and tails are what carry the comparison.
    plt.figure(figsize=(10, 6), layout='tight')
    ax = sns.kdeplot(data=real_dataset, x='velocity', color='#2a78d6', fill=True,
                     alpha=0.2, bw_adjust=BW, clip=CLIP, linewidth=3, label='Real (KABR)')
    sns.kdeplot(data=dataset, x='velocity', color='#eb6834', bw_adjust=BW, clip=CLIP,
                linewidth=3, label='Simulation (all params pooled)')
    ax.axvline(real_dataset['velocity'].mean(), color='#2a78d6', ls='--', lw=1.5)
    ax.axvline(dataset['velocity'].mean(), color='#eb6834', ls='--', lw=1.5)
    ax.set_xlim(0, XMAX_MS)
    ax.set_xlabel("Velocity (m/s)")

    # --- Similarity metrics (real vs pooled sim), same computation as curve_similarity
    # in compare_distributions.py: densities on a shared grid, then overlap/Pearson.
    real_v = real_dataset['velocity'].values
    sim_v = dataset['velocity'].values
    grid = np.linspace(0, XMAX_MS, 512)
    dx = grid[1] - grid[0]
    kde_r = stats.gaussian_kde(real_v, bw_method="scott"); kde_r.set_bandwidth(kde_r.factor * BW)
    kde_s = stats.gaussian_kde(sim_v, bw_method="scott"); kde_s.set_bandwidth(kde_s.factor * BW)
    fr, fs = kde_r(grid), kde_s(grid)
    fr /= fr.sum() * dx; fs /= fs.sum() * dx
    overlap = float(np.sum(np.minimum(fr, fs)) * dx)
    pearson = float(np.corrcoef(fr, fs)[0, 1])
    d_mean = dataset['velocity'].mean() - real_dataset['velocity'].mean()
    d_std = dataset['velocity'].std() - real_dataset['velocity'].std()
    txt = (f"Overlap   {overlap:.3f}\n"
           f"Pearson   {pearson:.3f}\n"
           f"Δ mean    {d_mean:+.3f} m/s\n"
           f"Δ SD      {d_std:+.3f} m/s")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.95))
    ax.legend(loc="center right")
    ax.get_figure().savefig(f"{output_directory}/velocity_real_vs_sim.pdf")

    print(f"Real:  N={len(real_dataset)}  mean={real_dataset['velocity'].mean():.3f} m/s")
    print(f"Sim :  N={len(dataset)}  mean={dataset['velocity'].mean():.3f} m/s")
    print(f"Overlap={overlap:.3f}  Pearson={pearson:.3f}  Δmean={d_mean:+.3f}  ΔSD={d_std:+.3f}")

    plt.show()
