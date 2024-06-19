import pandas as pd
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
from colorama import Fore
from colorama import Style
from matplotlib.colors import Colormap
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde
from IPython.display import display
from itertools import product


# Style Variables

FONT_COLOR = "#003440" #dark teal
BACKGROUND_COLOR = "#EDE9E0" # Silver

CELL_HOVER = {
    "selector": "td:hover", # <tr> for rows
    "props": "background-color: #cc0000", # Red
}

TEXT_HIGHLIGHT = { 
    "selector": "td",
    "props": "color: #E8E3DC; font-weight: light", # Silver
}

INDEX_NAMES = {
    "selector": ".index_name",
    "props": "font-style: italic; background-color: #003440; color: #E8E3DC;", # dark teal - Silver
}

HEADERS = {
    "selector": "th:not(.index_name)",
    "props": "font-style: italic; background-color: #003440; color: #E8E3DC;", # dark teal - Silver
}

# DataFrame Styling
DF_STYLE = (INDEX_NAMES, HEADERS, TEXT_HIGHLIGHT)

# For Dataframe Gradients
DF_CMAP: Colormap = sns.dark_palette("#00A5A8", as_cmap=True)
DF_CMAP2: Colormap = sns.dark_palette("#FF5508", as_cmap=True)
DF_CMAP_R: Colormap = sns.dark_palette("#00A5A8", as_cmap=True, reverse = True)
DF_CMAP2_R: Colormap = sns.dark_palette("#FF5508", as_cmap=True, reverse = True)
# For colorscales
color_map = [[0.0, "#00D8DB"], [0.5, "#002832"], [1.0, "#FF5508"]] # Cyan - Dark Blue - Orange

LIGHTEST_TEAL = "#05f9fc"
LIGHT_TEAL = "#00D8DB"
TEAL = "#00A5A8"
DARK_TEAL = "#003440"
DARKEST_TEAL = "#002832"

ORANGE = "#FF5508"

# Font Weights
#100: Thin
#200: Extra Light (Ultra Light)
#300: Light
#400: Normal (Regular)
#500: Medium
#600: Semi Bold (Demi Bold)
#700: Bold
#800: Extra Bold (Ultra Bold)
#900: Black (Heavy)










def numeric_description(data: pd.DataFrame, target_col: str, styling = False, gradient = False, gradient_axis = 1, float_precision = 3):

    numeric_descr = (
        data.drop(target_col, axis=1)
        .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .drop("count")
        .T.rename(columns=str.title)
    )

    if styling == False:
        return numeric_descr
    elif styling == True and gradient == False:
        return numeric_descr.style.set_table_styles(DF_STYLE).format(precision = float_precision)
    elif styling == True and gradient == True:
        return numeric_descr.style.set_table_styles(DF_STYLE).background_gradient(cmap=DF_CMAP, axis = gradient_axis).format(precision = float_precision)
    else:
        print("Invalid Inputs")










def correlation_values(data: pd.DataFrame, target: str, column_text_size = 10, width_value = 1000, height_value = 1000):
    pearson_corr = (
        data.drop(target, axis=1).corr(numeric_only=True, method="pearson").round(2)
    )
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    lower_triangular_corr = (
        pearson_corr.mask(mask)
        .dropna(axis="index", how="all")
        .dropna(axis="columns", how="all")
    )

    heatmap = go.Heatmap(
        z=lower_triangular_corr,
        x=lower_triangular_corr.columns,
        y=lower_triangular_corr.index,
        text=lower_triangular_corr.fillna(""),
        texttemplate="%{text}",
        xgap=1,
        ygap=1,
        showscale=True,
        colorscale=color_map,
        colorbar_len=1.02,
        hoverinfo="none",
    )
    fig = go.Figure(heatmap)
    fig.update_layout(
        font_color=FONT_COLOR,
        title="Correlation Matrix (Pearson) - Lower Triangular",
        title_font_size=18,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        width=width_value,
        height=height_value,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )
    fig.update_xaxes(tickfont=dict(size=column_text_size))
    fig.update_yaxes(tickfont=dict(size=column_text_size))
    fig.show()
    return pearson_corr, lower_triangular_corr










def dendogram(data: pd.DataFrame, target: str, pearson_corr):
    dissimilarity = 1 - np.abs(pearson_corr)

    fig = ff.create_dendrogram(
        dissimilarity,
        labels=pearson_corr.columns,
        orientation="left",
        colorscale=px.colors.sequential.YlGnBu_r,
        # squareform() returns lower triangular in compressed form - as 1D array.
        linkagefun=lambda x: linkage(squareform(dissimilarity), method="complete"),
    )
    fig.update_layout(
        font_color=FONT_COLOR,
        title="Hierarchical Clustering using Correlation Matrix (Pearson)",
        title_font_size=18,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        height=1340,
        width=840,
        yaxis=dict(
            showline=False,
            title="Feature",
            ticks="",
        ),
        xaxis=dict(
            showline=False,
            title="Distance",
            ticks="",
            range=[-0.05, 1.05],
        ),
    )
    fig.update_traces(line_width=1.5)
    fig.show()








def get_kde_estimation(data_series):
    kde = gaussian_kde(data_series.dropna())
    kde_range = np.linspace(
        data_series.min() - data_series.max() * 0.1,
        data_series.max() + data_series.max() * 0.1,
        len(data_series),
    )
    estimated_values = kde.evaluate(kde_range)
    estimated_values_cum = np.cumsum(estimated_values)
    estimated_values_cum /= estimated_values_cum.max()
    return kde_range, estimated_values, estimated_values_cum


def get_n_rows_axes(n_features, n_cols=5, n_rows=None):
    n_rows = int(np.ceil(n_features / n_cols))
    current_col = range(1, n_cols + 1)
    current_row = range(1, n_rows + 1)
    return n_rows, list(product(current_row, current_col))




def high_corr_combinations(correlation_threshold, lower_triangular_correlations):
    corr_threshold = correlation_threshold
    lower_triangular_corr = lower_triangular_correlations

    highest_abs_corr = (
        lower_triangular_corr.abs()
        .unstack()
        .sort_values(ascending=False)  # type: ignore
        .rename("Absolute Pearson Correlation")
    )

    highest_abs_corr = (
        highest_abs_corr[highest_abs_corr > corr_threshold]
        .to_frame()
        .reset_index(names=["Feature 1", "Feature 2"])
    )

    highest_corr_combinations = highest_abs_corr[["Feature 1", "Feature 2"]].to_numpy()
    display(highest_abs_corr.style.set_table_styles(DF_STYLE).format(precision=2))
    return highest_corr_combinations