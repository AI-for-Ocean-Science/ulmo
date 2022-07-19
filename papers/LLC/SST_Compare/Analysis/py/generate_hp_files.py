""" Generate the healpix files to speed up later aspects of the analysis """

from ulmo import io as ulmo_io

from ulmo.analysis import spatial_plots as sp

def generate_lonslats(table:str, out_root:str, write=False):

    # Load table
    tbl = ulmo_io.load_main_table(table)#'s3://viirs/Tables/VIIRS_all_98clear_std.parquet')

    # Heads
    head = tbl[ (tbl.datetime.dt.year > 2011) & (tbl.datetime.dt.year < 2015) ]

    # Evaluate
    evts_head, hp_lons_head, hp_lats_head, meds_head = sp.evals_to_healpix_meds(
        eval_tbl=head, nside=64,  mask=True)

    # Tails
    tail = tbl[ (tbl.datetime.dt.year > 2017) & (tbl.datetime.dt.year < 2021) ]
    evts_tail, hp_lons_tail, hp_lats_tail, meds_tail = sp.evals_to_healpix_meds(eval_tbl=tail, nside=64,  mask=True)

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


# Command line execution
if __name__ == '__main__':
    generate_lonslats('s3://viirs/Tables/VIIRS_all_98clear_std.parquet',
                      '_v98', write=True)

