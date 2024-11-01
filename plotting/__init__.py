import seaborn as sbn
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot as plt

# ========================================
# Matplotlib settings
params = {
    "legend.fontsize": 16,
    "legend.frameon": False,
    "ytick.labelsize": 15,
    "xtick.labelsize": 15,
    # "figure.dpi": 300, # 600,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    # "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
    # "font.sans-serif": "Arial",
    # "font.family": "sans-serif",
    # "font.weight": "bold",
    "axes.unicode_minus": False,
}
plt.rcParams.update(params)

# ========================================
# Customized color palettes
dark_gray_color = (50 / 255, 50 / 255, 50 / 255)
gray_color = (173 / 255, 181 / 255, 189 / 255)
light_gray_color = (192 / 255, 192 / 255, 192 / 255)
white_color = (1.0, 1.0, 1.0)

from palettable.tableau import BlueRed_12, Tableau_20, Tableau_10
from palettable.cmocean.sequential import Thermal_12
from palettable.cartocolors.qualitative import Vivid_10, Bold_10
from palettable.matplotlib import Magma_20, Viridis_20, Inferno_10, Inferno_6
from palettable.mycarta import Cube1_20, Cube1_6, Cube1_12

Kelly20 = [
    "#ebce2b", "#702c8c", "#db6917", "#96cde6", "#ba1c30",
    "#c0bd7f", "#7f7e80", "#5fa641", "#d485b2", "#4277b6",
    "#df8461", "#463397", "#e1a11a", "#91218c", "#e8e948",
    "#7e1510", "#92ae31", "#6f340d", "#d32b1e", "#2b3514"
]

model_color = {
    "scMultiNODE": Tableau_10.mpl_colors[3],
    "SCOTv2": Tableau_10.mpl_colors[0],
    "SCOTv1": Tableau_10.mpl_colors[1],
    "Pamona": Tableau_10.mpl_colors[2],
    "UnionCom": Tableau_10.mpl_colors[4],
    "uniPort": Tableau_10.mpl_colors[5],
    "Seurat": Tableau_10.mpl_colors[7],
}

# ========================================

def linearSegmentCMap(num_colors, cmap_name="viridis"):
    '''Construct colormap for linearly segmented colors.'''
    cm = plt.get_cmap(cmap_name)
    color_list = [cm(i//3*3.0/num_colors) for i in range(num_colors)]
    return color_list


def removeTopRightBorders(ax=None):
    '''Remove top and right borders of the figure.'''
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def removeAllBorders(ax=None):
    '''Remove all borders of the figure.'''
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)