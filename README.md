# PaperPlots #

A python script that uses matplotlib to generate plots.
It create lines graphs of the averages of the datasets with optional standard deviation bars above and below.
There are a number of optional and required parameters which are listed below.

## Options ##

The command line options are as follows:

   - `--infiles` A space delimited list of row-major CSV files
   - `--outfile` The name of the output file (e.g. `plot.png`, this argument is optional)
   - `--title` The title (optional)
   - `--xs` The labels on the x-axis (e.g. `4:31` means "four to thirty-one", this argument is optional)
   - `--xlabel` The x-axis label (optional)
   - `--ylabel` The y-axis label (optional)
   - `--most` The maximum y value (optional)
   - `--least` The minimum y value (optional)
   - `--aspect` The desired aspect ratio of the plot (optional)
   - `--names` The names of the datasets; should correspond 1:1 with the `--infiles` list
   - `--mu` The colors associated with the respective average values of the datasets (e.g. `#0000ff`); should correspond 1:1 with the `--infiles` list
   - `--sigma` The colors associated with the respective standard deviations of the datasets (e.g. `#ff0000`); the length of this list must be ≤ that of the list given to `--infiles`
   
