import matplotlib as mpl

name = "Default settings for Matplotlib"

cm2inch = lambda x: 0.393700787 * x


#for plos
fig_width = dict(
    one_col = cm2inch(7.),
    two_col = cm2inch(16.)
    )

font_panellabel = dict(fontweight='bold', fontsize=12, ha='left')

config = {
    'axes' : dict(labelsize=6, titlesize=6, linewidth=0.5, labelpad=2.),
    'figure' : dict(dpi=109, figsize=[fig_width['two_col'], 0.75*fig_width['two_col']], facecolor='white'),
    # 'figure' : dict(dpi=86.4, figsize=[fig_width['two_col'], 0.75*fig_width['two_col']], facecolor='white'),
    'figure.subplot' : dict(left=0.15, bottom=0.15, right=0.97, top=0.97),
    'font' : {'family' : 'sans-serif', 'size' : 8, 'weight' : 'normal',
              'sans-serif' : ['Arial', 'LiberationSans-Regular', 'FreeSans']},
    'image' : dict(cmap='RdBu_r' , interpolation='nearest'),
    'legend' : dict(fontsize=8, borderaxespad=0.5, borderpad=0.5),
    'lines' : dict(linewidth=0.5),
    'xtick' : dict(labelsize=6),
    'xtick.major' : dict(size=2, pad=2, width=0.5),
    'ytick' : dict(labelsize=6),
    'ytick.major' : dict(size=2, pad=2, width=0.5),
    'savefig' : dict(dpi=600)
    }

print ("\n\t * * * Importing '%s' * * *\n" % name)

for key,val in config.items():
    s = ""
    for k,v in val.items():
        s += k + "=%s, " % str(v)
    print ("  > Set '%s' to %s" % (key, s[:-2]) )
    mpl.rc(key, **val)

print ('\n\n')
