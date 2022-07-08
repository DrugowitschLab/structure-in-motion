"""
Definitions of colors and color maps.
"""
import numpy as np
import pylab as pl


CMAP = {
    "glo" : pl.cm.spring_r,
    "clu" : pl.cm.Blues,
    "ind" : pl.cm.Greens,
    "velo": pl.cm.Oranges,
    "lin" : pl.cm.cool,
    "one" : pl.cm.bone,
}

SINGLE = {
    "self" : "#eacc00ff" ,
    "vestibular" : "#f9715fff",
    "rotation" : "#E60080ff",
    "rotation_light" : "#FF4DCDff"
}


def get_colors(cmap, N):
    return cmap(np.linspace(0.90, 0.4, N))
    # return cmap(np.linspace(0.85, 0.2, N))     # for "one"
    # return cmap(np.linspace(0.65, 0.95, N))  # for "lin"

def get_color(cmap, i, N):
    return get_colors(cmap, N)[i]




if __name__ == "__main__":
    N = 3
    cmap = CMAP["velo"]
    
    fig = pl.figure(figsize=(6,2))
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.92, 0.25, 0.25)
    ax_cm = fig.add_subplot(2,1,1, xticks=[], yticks=[])
    axes = [ fig.add_subplot(2,N,N+i+1, xticks=[], yticks=[], aspect="equal") for i in range(N) ]
    
    # gradient
    gradient = np.linspace(1, 0, 256)
    gradient = np.vstack((gradient, gradient))
    ax_cm.imshow(gradient, aspect='auto', cmap=cmap)
    ax_cm.set_title(f"Colormap '{cmap.name}'", pad=3)

    # single object colors
    for i in range(N):
        c = get_color(cmap, i, N)
        cstring = pl.matplotlib.colors.rgb2hex(c, keep_alpha=False)
        cstringlong = pl.matplotlib.colors.rgb2hex(c, keep_alpha=True)
        axes[i].set_facecolor(c)
        axes[i].set_title(cstring, pad=3)
        print(f" > Color for object {i+1} of {N}: {cstringlong[1:]}")
