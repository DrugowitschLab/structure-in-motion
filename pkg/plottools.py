"""
General helper functions and constants for plotting.
Written by Johannes Bill.
"""

import numpy as np
import pylab as pl

def set_zoom(factor):
    """Change the zoom factor for figures on the screen (does not affect saved figure size)."""
    from matplotlib import rc
    rc("figure", dpi=factor * pl.matplotlib.rcParams["figure.dpi"])


def init_figure_absolute_units(figsize, axes, units="inches", figureKwargs={}):
    """
    Set up a figure with several axes such that all coordinates are given in absolute size (inches or cm).
    
    figsize = (width, height)
              --> width, height given in inches or cm
    axes is a dict with
        axes[axesname] = dict(rect=(left, top, width, height), [optional]f_callafter, **kwargs)
                         --> (left, top, width, height) in units inches or cm
                         --> top is top-edge measured from top of figure
                         --> If present, f_callafter(ax) will be called after axes creation (e.g., set label padding)
                         --> kwargs will be passed to fig.add_axes()
                         
    Returns tuple (fig, axesdict) with axesdict[axesname] = ax
    """
    # Set correct transformations
    if units == "inches":
        f_transform = lambda x: np.array(x)
    elif units == "cm":
        f_transform = lambda x: 0.393701 * np.array(x)
    else:
        raise Exception(f"Unknown units '{units}'.")
    
    # Make figure
    fig = pl.figure(figsize=f_transform(figsize), **figureKwargs)
    
    # Little helper for transforming absolute to relative coords
    # (see: https://discourse.matplotlib.org/t/add-axes-in-inches/12509/2)
    def relative_rect(rect_inches):
        from matplotlib.transforms import Bbox, BboxTransformFrom, TransformedBbox
        tr = BboxTransformFrom(Bbox.from_bounds(0, 0, *fig.get_size_inches()))
        rect = TransformedBbox(Bbox.from_bounds(*rect_inches), tr).bounds
        return list(rect)
        
    # Add axes
    axesout = dict()
    for axname, axdict in axes.items():
        rect = relative_rect(f_transform(axdict.pop('rect')))
        rect[1] = 1. - (rect[1] + rect[3])          # from top to bottom
        f_callafter = axdict.pop('f_callafter', lambda ax: None)
        ax = fig.add_axes(rect, **axdict)
        f_callafter(ax)
        axesout[axname] = ax
        
    return fig, axesout


def print_panel_label_abcolute_units(ax, label, x, y, units="inches", fontkwargs={}):
    # Set correct transformations
    if units == "inches":
        f_transform = lambda x: np.array(x)
    elif units == "cm":
        f_transform = lambda x: 0.393701 * np.array(x)
    else:
        raise Exception(f"Unknown units '{units}'.")
        
    fig = ax.get_figure()
    
    # Little helper for transforming absolute to relative coords
    # (see: https://discourse.matplotlib.org/t/add-axes-in-inches/12509/2)
    def relative_rect(rect_inches):
        from matplotlib.transforms import Bbox, BboxTransformFrom, TransformedBbox
        tr = BboxTransformFrom(Bbox.from_bounds(0, 0, *fig.get_size_inches()))
        rect = TransformedBbox(Bbox.from_bounds(*rect_inches), tr).bounds
        return list(rect)

    rect = 0., 0., x, y
    rect = relative_rect(f_transform(rect))
    rect[1] = 1. - (rect[1] + rect[3])          # from top to bottom
    x, y = rect[2], rect[1]
    
    kwargs = dict(fontsize=8., fontweight="bold", ha="left", va="baseline", clip_on=False, transform=fig.transFigure)
    kwargs.update(fontkwargs)
    text = ax.text(x, y, label, **kwargs)
    return text


def background_patch_absolute_units(ax, rect, color="peachpuff", units="inches"):
    """
    Create a patch for background coloring.
    
    ax    : the Axes to draw into
    rect  : (left, top, width, height) with top being top-edge measured from top of the figure
    color : any matplotlib color
    units : inches or cm
    
    Returns:
      Patch object
    """
    # Set correct transformations
    if units == "inches":
        f_transform = lambda x: np.array(x)
    elif units == "cm":
        f_transform = lambda x: 0.393701 * np.array(x)
    else:
        raise Exception(f"Unknown units '{units}'.")
        
    fig = ax.get_figure()
    
    # Little helper for transforming absolute to relative coords
    # (see: https://discourse.matplotlib.org/t/add-axes-in-inches/12509/2)
    def relative_rect(rect_inches):
        from matplotlib.transforms import Bbox, BboxTransformFrom, TransformedBbox
        tr = BboxTransformFrom(Bbox.from_bounds(0, 0, *fig.get_size_inches()))
        rect = TransformedBbox(Bbox.from_bounds(*rect_inches), tr).bounds
        return list(rect)
    
    rect = relative_rect(f_transform(rect))
    rect[1] = 1. - (rect[1] + rect[3])          # from top to bottom

    # We adjust the rounding_size by the figure width to obtain size-independent shapes
    rounding_size = 0.008 * 7. / fig.get_size_inches()[0]
    FancyBboxPatch = pl.matplotlib.patches.FancyBboxPatch
    kwargs = dict(fc=color, ec=None, lw=0., transform=fig.transFigure)
    kwargs["mutation_aspect"] = fig.get_size_inches()[0] / fig.get_size_inches()[1] # make corner radius symmetric
    boxstyle = f"round,pad=0.0,rounding_size={rounding_size:.4f}"
    patch = FancyBboxPatch((rect[0], rect[1]), rect[2], rect[3], boxstyle=boxstyle, **kwargs)
    ax.add_patch(patch)
    return patch



def savefig(fig, basename, path="./fig/", fmt=(".png", ".pdf"), dpi=600, logger=None):
    """
    Save figure fig in all formats listed in fmt.
    If path does not exist, it will be created. Existing files are overwritten.
    """
    from pathlib import Path
    p = Path(path)
    # Make target dir
    p.mkdir(parents=True, exist_ok=True)
    # check basepath
    assert isinstance(basename, str) and ("/" not in basename) and (basename[-1]  != ".")
    for ftype in fmt:
        kwargs = {"dpi" : dpi if ftype in (".png", ".jpg", ".jpeg") else None}
        fname = p / (basename + ftype)
        s = f"Save figure to file '{fname}'."
        if logger:
            logger.info(s)
        else:
            print(s)
        fig.savefig(fname, dpi=dpi)
    
    
def auto_axes_lim(ax, which=("x", "y"), ymin=None, ymargin=0.07):
    """
    Auto-adjust the xlim and ylim of axes 'ax'.
    
    Optional:
        which : which axes to adjust
        ymin  : if given, sets ymin. Otherwise ymin is auto-adjusted
        ymargin : relative margin beyond data range
    """
    if ("x" in which) or (which == "x"):
        ax.set_xlim(ax.dataLim.xmin, ax.dataLim.xmax)
    if ("y" in which) or (which == "y"):
        y1 = ax.dataLim.ymax
        y0 = ymin if (ymin is not None) else ax.dataLim.ymin
        d = ymargin * (y1 - y0)
        y1 += d
        if ymin is None:
            y0 -= d
        ax.set_ylim(y0, y1)
        
        
def cmap_alpha(color, alphamin=0.0, alphamax=1.0):
    """Make a simple colormap with a single color, but varying opacity -- ranging from alphamin to alphamax."""
    LSC = pl.matplotlib.colors.LinearSegmentedColormap
    c1 = pl.matplotlib.colors.to_rgba(color, alpha=alphamax)
    c0 = tuple(c1)[:3] + (alphamin,)
    return LSC.from_list(f"alpha_{color}", (c0, c1))

def cmap_white(color):
    """Make a simple colormap from white to a single color"""
    LSC = pl.matplotlib.colors.LinearSegmentedColormap
    c1 = pl.matplotlib.colors.to_rgba(color, alpha=1.0)
    c0 = (1.,1.,1.,1.)
    return LSC.from_list(f"white_to_{color}", (c0, c1))

