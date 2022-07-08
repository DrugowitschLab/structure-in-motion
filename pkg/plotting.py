"""Various helpers for plotting."""

import numpy as np
import pylab as pl

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Crop a piece from a cmap."""
    # From https://stackoverflow.com/a/18926541
    new_cmap = pl.matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def find_groups(cnorm, groups=None, idxlist=None):
    """Group components by their norm, recursively."""
    if groups is None:
        groups = []   # Do not use default arg for repeated application
    if idxlist is None:
        idxlist = np.arange(len(cnorm))
    idx = np.where(cnorm == cnorm[0])[0]
    groups.append(list(idxlist[idx]))
    cnorm = np.delete(cnorm, idx)
    idxlist = np.delete(idxlist, idx)
    if len(cnorm) == 0:
        return groups
    else:
        groups = find_groups(cnorm, groups=groups, idxlist=idxlist)
        return groups

from .colors import CMAP
# This is the default coloring for the motion features
groupColor = ( CMAP["glo"], CMAP["clu"], CMAP["ind"], pl.cm.copper, pl.cm.pink, pl.cm.Greys )



def assign_colors(C):
    """Try to auto-assign colors to motion components."""
    cnorm = np.abs(C).sum(0)  # Manhattan norm
    groups = find_groups(cnorm)
    colors = [ c for i,gi in enumerate(groups) for c in groupColor[i % len(groupColor)](np.linspace(0.85, 0.4, len(gi))) ]
    colors = np.array(colors).reshape(-1,4)
    return colors


def imshow_C_matrix(C, ax, colors=None, addVisibleCircles=False, viscolors=None):
    """Plot matrix C into axes ax (usually an inset)."""
    K,M = C.shape
    if colors is None:
        colors = assign_colors(C)
    from matplotlib.colors import LinearSegmentedColormap as LSC
    for m in range(M):
        Cm = C[:,m:m+1]
        if Cm.min() == Cm.max():
            cmap = LSC.from_list("tmp", (colors[m], colors[m]))
        elif -1 in Cm:
            cmap = LSC.from_list("tmp", (colors[m], 'w', colors[m]))
        else:
            cmap = LSC.from_list("tmp", ('w', colors[m]))
        extent = (m-0.5, m+0.5, K-0.5, -0.5)
        ax.imshow(Cm, cmap=cmap, extent=extent, interpolation="nearest")
        # counter rotating?
        if -1 in Cm:
            for k,ckm in enumerate(Cm[:,0]):
                s = {-1 : "-", +1 : "+", 0 : ""}[ckm]
                ax.text(m, k, s, va="center", ha="center", size=5)
    ax.set_xlim(-0.5, M-0.5)
    ax.set_ylim(K-0.5, -0.5)
    # Circles indicating the visibles
    if not addVisibleCircles:
        return
    assert (viscolors is not None) and (len(viscolors) == K)
    x = -1.15
    y = np.arange(K)
    kwargs = dict(lw=0., ec=None, fill=True, clip_on=False)
    for yk,ck in zip(y, viscolors): 
        p = pl.matplotlib.patches.Circle((x,yk), radius=0.85/2, fc=ck, **kwargs)
        ax.add_patch(p)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # #  THE FOLLOWING ARE JUST FOR INITIAL PLOTTING RIGHT AT THE END OF THE SIMULATION   # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def plot_inferred_lambda(t, Lam, C, lam_star=None, cfg=None, colors=None, ax=None, axC=None):
    """
    Plot lambda(t) for all filters in dict Lam["filter name"] = array lam.
    Colors are determined from component matrix C.
    Optionally, the ground truth lam_star can be shown (can be a lam_list).
    If cfg dictionary is provided, some additional info is plotted in the title.
    """
    K,M = C.shape
    if colors is None:
        colors = assign_colors(C)

    if ax is not None:
        assert axC is not None 
        fig = ax.get_figure()
    else:
        fig = pl.figure(figsize=(7,2.5))
        # Component legend
        w, h = 0.20 * M / K * fig.get_figheight()/fig.get_figwidth() , 0.20
        rect = 0.08, 0.90-h, w, h
        axC = fig.add_axes(rect, aspect="equal", frame_on=True, xticks=[], yticks=[], zorder=10)
    from matplotlib.colors import LinearSegmentedColormap as LSC
    for m in range(M):
        Cm = C[:,m:m+1]
        if Cm.min() == Cm.max():
            cmap = LSC.from_list("tmp", (colors[m], colors[m]))
        elif -1 in Cm:
            cmap = LSC.from_list("tmp", (colors[m], 'w', colors[m]))
        else:
            cmap = LSC.from_list("tmp", ('w', colors[m]))
        extent = (m-0.5, m+0.5, K-0.5, -0.5)
        axC.imshow(Cm, cmap=cmap, extent=extent, interpolation="nearest")
        # counter rotating?
        if -1 in Cm:
            for k,ckm in enumerate(Cm[:,0]):
                s = {-1 : "-", +1 : "+", 0 : ""}[ckm]
                axC.text(m, k, s, va="center", ha="center", size=5)
    axC.set_xlim(-0.5, M-0.5)
    axC.set_ylim(K-0.5, -0.5)
    # Lambda (t)
    if ax is None:
        rect = 0.07, 0.12, 0.98-0.07, 0.92-0.12
        ax = fig.add_axes(rect, zorder=1)
    # linestyle = ("-", (0, (5, 4)) , (0, (3, 2, 1, 4)), ":", "--", "-.")
    linestyle = ((0, (5, 4)), "-" , (0, (3, 2, 1, 4)), ":", "--", "-.")
    ymax = 0.
    handles = []
    for i, (filname,lam) in enumerate(Lam.items()):
        ymax = max(ymax, lam.max())
        ax.set_prop_cycle(c=colors)
        lines = ax.plot(t, lam, ls=linestyle[i], label=filname.capitalize())
        handles.append(lines[0])
    # lam_star
    if lam_star is not None:
        lam_star = np.array(lam_star, dtype='object')
        if lam_star.ndim == 1:
            # lam_star = np.array([ (0., lam_star) ])
            lam_star = [ (0., lam_star) ]
        lmax = max([max(l[1]) for l in lam_star])
        ymax = max(ymax, lmax)
        for n,(t1,lam) in enumerate(lam_star):
            ax.vlines([t1], 0, 1.5*lmax, linestyles=(0, (1,2)), colors=['0.3'])
            try:
                t2 = lam_star[n+1][0]
            except:
                t2 = t[-1]
            label = "Ground truth"  if n == 0 else None
            lines = ax.hlines(lam, t1, t2, linestyles=(0, (2,2)), colors=colors, label=label)
            if n == 0:
                handles.append(lines)
    ax.set_xlabel("Time [s]", labelpad=1)
    ax.set_xlim(t[0],t[-1])
    ax.set_ylabel(r"Inferred motion strengths $\lambda_m$")
    ax.set_ylim(0.0, 1.05*ymax)
    leg = ax.legend(handles=handles, loc="upper right")
    # Title
    if cfg is not None:
        items = [ ( "sig_obs" , cfg["fil"]["default_params"], r"$\sigma_\mathrm{obs}$" ),
                  ( "tau_s"   , cfg["fil"]["default_params"], r"$\tau_s$" ),
                  ( "tau_lam" , cfg["fil"]["default_params"]["inf_method_kwargs"], r"$\tau_\lambda$" ),
                  ( "kappa"   , cfg["fil"]["default_params"]["inf_method_kwargs"], r"$\kappa$" ),
                  ( "nu"      , cfg["fil"]["default_params"]["inf_method_kwargs"], r"$\nu$" ),
                ]
        titlestr = ""
        for k,d,s in items:
            if k in d:
                v = d[k]
                if isinstance(v, np.ndarray):
                    if (v[0] == v).all():
                        v = v[0]
                        titlestr += f"{s}={v:.3f},  "
                    else:
                        # v = "diverse"
                        titlestr += f"{s}={v},  "
                    continue
                titlestr += f"{s}={v:.3f},  "
        titlestr = titlestr[:-3]
        ax.set_title( titlestr, pad=3)
    return dict(fig=fig, axes=(ax, axC), leg=leg, color=colors)


def plot_inferred_source(t, S, C, S_star=None, colors=None, ax=None):
    """
    Plot <s>(t) for all filters in dict S["filter name"] = array lam.
    Colors are determined from component matrix C.
    Optionally, the ground truth S_star can be shown.
    """
    K,M = C.shape
    if colors is None:
        colors = assign_colors(C)
    
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig = pl.figure(figsize=(7,2.5))
        # S (t)
        rect = 0.07, 0.12, 0.98-0.07, 0.92-0.12
        ax = fig.add_axes(rect, zorder=1)
    # linestyle = ("-", (0, (5, 4)) , (0, (3, 2, 1, 4)), ":", "--", "-.")
    linestyle = ((0, (5, 4)), "-" , (0, (3, 2, 1, 4)), ":", "--", "-.")
    ymax = 0.
    handles = []
    for i, (filname,s) in enumerate(S.items()):
        ymax = max(ymax, np.abs(s).max())
        ax.set_prop_cycle(c=colors)
        lines = ax.plot(t, s, ls=linestyle[i], label=filname.capitalize())
        handles.append(lines[0])
    # S_star
    if S_star is not None:
        s, filname = S_star, "Ground truth"
        ymax = max(ymax, np.abs(s).max())
        ax.set_prop_cycle(c=colors)
        lines = ax.plot(t, s, ls=(0, (2,2)), label=filname)
        handles.append(lines[0])
    ax.set_xlabel("Time [s]", labelpad=1)
    ax.set_xlim(t[0],t[-1])
    ax.set_ylabel(r"Motion sources $s_m(t)$")
    ax.set_ylim(-1.05*ymax, 1.05*ymax)
    leg = ax.legend(handles=handles, loc="upper right")
    return dict(fig=fig, axes=ax, leg=leg, color=colors)



