"""  Module for Tables for the SSL paper """
# Imports
import os, sys


#from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import ssl_paper_analy

# ulmo needs to come after the above!
from ulmo.utils import catalog

def mktab_cutouts(outfile='tab_cutouts.tex', sub=False, local=True):

    if sub:
        outfile=outfile.replace('.tex', '_sub.tex')

    # Load up 
    tbl_dict = {}


    # All the DT tables too!
    for subset in ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5', 'DTall']:
        tbl_dict[subset] = ssl_paper_analy.load_modis_tbl(
            local=local, region=None, table=f'96clear_v4_{subset}',
            percentiles=None)
        # Fill in DT subsets
        if subset == 'DTall':
            # Sort by date
            tbl_dict[subset].sort_values('datetime', inplace=True)
            # Init
            tbl_dict[subset]['SU0'] = 0.
            tbl_dict[subset]['SU1'] = 0.
            for key in ['DT15', 'DT0', 'DT1', 'DT2', 'DT4', 'DT5']:
                # Match em
                idx = catalog.match_ids(tbl_dict[key].UID.values, 
                                        tbl_dict[subset].UID.values)
                # Fill in
                tbl_dict[subset].SU0.values[idx] = tbl_dict[key].U0.values
                tbl_dict[subset].SU1.values[idx] = tbl_dict[key].U1.values

    

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{MODIS Cutouts\\label{tab:cutouts}}\n')
    tbfil.write('\\begin{tabular}{cccccccccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('lon & lat & date & $\\Delta T$ & \\slope & LL & $U_{0,\\rm all}$ & $U_{1,\\rm all}$')
    tbfil.write('& $U_0$ & $U_1$ \\\\ \n')
    tbfil.write('(deg) & (deg) & & (K) \n') 
    tbfil.write('\\\\ \n')
    tbfil.write('\\hline \n')

    # Loop me 
    count = 0
    for index, row in tbl_dict['DTall'].iterrows():
        if sub and count > 20:
            break
        count += 1

        # Lat, lon
        slin = f'{row.lon:0.3f} & {row.lat:0.3f}'

        # Date
        slin += f'& {row.datetime}'

        # DT 40
        slin += f'& {row.DT40:0.3f}'

        # slope
        slin += f'& {row.min_slope:0.2f}'

        # Ulmo LL
        slin += f'& {row.LL:0.1f}'

        # U0, U1
        slin += f'& {row.U0:0.1f} & {row.U1:0.1f}'

        # U0, U1 for DT subset
        slin += f'& {row.SU0:0.1f} & {row.SU1:0.1f}'

        tbfil.write(slin)
        tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    #tbfil.write('\\end{minipage} \n')
    tbfil.write('\\\\ \n')
    tbfil.write('Notes: The \\DT\\ value listed here is measured from the inner $40 \\times 40$\,pixel$^2$ region of the cutout. \\\\ \n')
    tbfil.write('LL is the log-likelihood metric calculated from the \\ulmo\\ algorithm. \\\\ \n')
    tbfil.write('$U_{0,all}, U_{1,all}$ are the UMAP values for the UMAP analysis on the full dataset. \\\\ \n')
    tbfil.write('$U_0, U_1$ are the UMAP values for the UMAP analysis in the \\DT\\ bin for this cutout. \\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))



# Command line execution
if __name__ == '__main__':

    mktab_cutouts(sub=True)