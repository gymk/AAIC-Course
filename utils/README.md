
to_pdf.sh
- Based on hoogenm's suggestion in https://github.com/jupyter/nbconvert/issues/524 for converting Python notebook to PDF without line truncation.

To know Font cache folder location

- method 1
```
import matplotlib as mpl
print(mpl.font_manager.get_cachedir())
```

Better images in PDF

- configure to have PDF instead of PNG while plotting images
```
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.figsize'] = 10,6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
```

- additional if possible use latex
```
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cm'
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"
```
