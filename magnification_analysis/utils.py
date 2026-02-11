

def setup_journal_style():
    """Configure matplotlib for Medical Image Analysis journal standards."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'legend.fontsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'text.usetex': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })



