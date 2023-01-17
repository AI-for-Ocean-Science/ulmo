""" Generate the healpix files to speed up later aspects of the analysis """

from ulmo import io as ulmo_io

from ulmo.analysis import spatial_plots as sp

#sys.path.append(os.path.abspath("../Analysis/py"))
import sst_compare_utils

def generate_headtails(table:str, out_root:str, write=False):

    # Load table
    tbl = ulmo_io.load_main_table(table)#'s3://viirs/Tables/VIIRS_all_98clear_std.parquet')

    # Heads
    head = tbl[ (tbl.datetime.dt.year > 2011) & (tbl.datetime.dt.year < 2015) ]

    # Evaluate
    evts_head, hp_lons_head, hp_lats_head, meds_head = sp.evals_to_healpix_meds(
        eval_tbl=head, nside=64,  mask=True)

    # Tails
    tail = tbl[ (tbl.datetime.dt.year > 2017) & (tbl.datetime.dt.year < 2021) ]
    evts_tail, hp_lons_tail, hp_lats_tail, meds_tail = sp.evals_to_healpix_meds(
        eval_tbl=tail, nside=64,  mask=True)

    # Write
    if write:
        evts_head.dump('evts_head'+out_root)
        hp_lons_head.dump('hp_lons_head'+out_root)
        hp_lats_head.dump('hp_lats_head'+out_root)
        meds_head.dump('meds_head'+out_root)
        # Tails
        evts_tail.dump('evts_tail'+out_root)
        hp_lons_tail.dump('hp_lons_tail'+out_root)
        hp_lats_tail.dump('hp_lats_tail'+out_root)
        meds_tail.dump('meds_tail'+out_root)

def generate_all(dataset:str, out_root:str, 
                 cut_DT:tuple=None, local:bool=False):

    # Load table
    tbl = sst_compare_utils.load_table(dataset, local=local,
                                       cut_DT=cut_DT)
    print(f"We have {len(tbl)} cutouts satisfying the cuts")

    # Evaluate
    evts, hp_lons, hp_lats, meds= sp.evals_to_healpix_meds(
        eval_tbl=tbl, nside=64,  mask=True)
    print("Done evaluating")

    # Write
    evts.dump('evts'+out_root)
    hp_lons.dump('hp_lons'+out_root)
    hp_lats.dump('hp_lats'+out_root)
    meds.dump('meds'+out_root)

# Command line execution
if __name__ == '__main__':

    # Heads and tails for VIIRS
    #generate_headtails('s3://viirs/Tables/VIIRS_all_98clear_std.parquet',
    #                  '_v98', write=True)

    # All for VIIRS
    #generate_all('viirs', '_v98')

    # All for LLC Uniform
    #generate_all('llc_uniform', '_llc_uniform')

    # All for LLC matched
    #generate_all('llc_match', '_llc_match')

    # MODIS, unmatched
    #generate_all('modis_all', '_modis_all')


    # DT = 1-1.X for VIIRS
    #generate_all('viirs', '_v98_DT112', cut_DT=(1.,1.2), local=True)
    #generate_all('viirs', '_v98_DT115', cut_DT=(1.,1.5), local=True)
    pass