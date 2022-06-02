"""  Module for Tables for the SSL paper """
# Imports
import numpy as np
import os, sys
import pandas

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import ssl_paper_analy

def mktab_cutouts(outfile='tab_cutouts.tex', sub=False, local=True):

    if sub:
        outfile=outfile.replace('.tex', '_sub.tex')

    # Load up 
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, cuts=None, 
        region=None, table='CF',
        percentiles=None)

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{MODIS Cutouts\\label{tab:cutouts}}\n')
    tbfil.write('\\begin{tabular}{cccccccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('lon & lat & date & $\\Delta T$ & \\slope & LL & U0 & U1 \n')
    tbfil.write('\\\\ \n')
    tbfil.write('\\hline \n')

    # Loop me 
    count = 0
    for index, row in modis_tbl.iterrows():
        if sub and count > 20:
            break
        count += 1

        # Lat, lon
        slin = f'{row.lon:0.3f} & {row.lat:0.3f}'

        # Date
        slin += f'& {row.datetime}'

        # DT 
        slin += f'& {row.DT:0.3f}'

        # slope
        slin += f'& {row.min_slope:0.2f}'

        # Ulmo LL
        slin += f'& {row.LL:0.1f}'

        # U0, U1
        slin += f'& {row.U0:0.1f} & {row.U1:0.1f}'

        tbfil.write(slin)
        tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    #tbfil.write('\\end{minipage} \n')
    #tbfil.write('{$^a$}Rest-frame value.  Error is dominated by uncertainty in $n_e$.\\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))



# Command line execution
if __name__ == '__main__':

    mktab_cutouts(sub=True)