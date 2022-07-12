import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_library_sizes(adata_v9, ct_key='Broad cell type'):
    library_sizes = adata_v9.X.toarray().sum(axis=-1)
    df = pd.DataFrame(
        {'Cell type': adata_v9.obs[ct_key], 'Tissue': adata_v9.obs['Tissue'], 'Library size': library_sizes})

    sns.violinplot(y='Library size', x='Cell type', data=df)
    plt.xticks(rotation=90)
    return plt.gca()


#### ENRICHMENT BARPLOT
def enr_barplot(df, column='Adjusted P-value', title="", cutoff=0.05, top_term=10,
                figsize=(6.5, 6), color='salmon', ofname=None, ax=None, **kwargs):
    """Visualize enrichr results.
    :param df: GSEApy DataFrame results.
    :param column: which column of DataFrame to show. Default: Adjusted P-value
    :param title: figure title.
    :param cutoff: terms with 'column' value < cut-off are shown.
    :param top_term: number of top enriched terms to show.
    :param figsize: tuple, matplotlib figsize.
    :param color: color for bars.
    :param ofname: output file name. If None, don't save figure

    """

    from matplotlib.ticker import MaxNLocator

    def isfloat(x):
        try:
            float(x)
        except:
            return False
        else:
            return True

    def adjust_spines(ax, spines):
        """function for removing spines and ticks.
        :param ax: axes object
        :param spines: a list of spines names to keep. e.g [left, right, top, bottom]
                        if spines = []. remove all spines and ticks.
        """
        for loc, spine in ax.spines.items():
            if loc in spines:
                # spine.set_position(('outward', 10))  # outward by 10 points
                # spine.set_smart_bounds(True)
                continue
            else:
                spine.set_color('none')  # don't draw spine

        # turn off ticks where there is no spine
        if 'left' in spines:
            ax.yaxis.set_ticks_position('left')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])

    colname = column
    if colname in ['Adjusted P-value', 'P-value']:
        # check if any values in `df[colname]` can't be coerced to floats
        can_be_coerced = df[colname].map(isfloat)
        if np.sum(~can_be_coerced) > 0:
            raise ValueError('some value in %s could not be typecast to `float`' % colname)
        else:
            df.loc[:, colname] = df[colname].map(float)
        df = df[df[colname] <= cutoff]
        if len(df) < 1:
            msg = "Warning: No enrich terms using library %s when cutoff = %s" % (title, cutoff)
            return msg
        df = df.assign(logAP=lambda x: - x[colname].apply(np.log10))
        colname = 'logAP'

    dd = df.sort_values(by=colname).iloc[-top_term:, :]
    # create bar plot

    if ax is None:
        ax = fig.add_subplot(111)
    bar = dd.plot.barh(x='Term', y=colname, color=color,
                       alpha=0.75, fontsize=12, ax=ax)

    if column in ['Adjusted P-value', 'P-value']:
        xlabel = "-log$_{10}$(%s)" % column
    else:
        xlabel = column
    bar.set_xlabel(xlabel, fontsize=12)
    bar.set_ylabel("")
    bar.set_title(title, fontsize=12, fontweight='bold')
    bar.xaxis.set_major_locator(MaxNLocator(integer=True))
    bar.legend_.remove()
    adjust_spines(ax, spines=['left', 'bottom'])

    if ofname is None:
        return
    return ax