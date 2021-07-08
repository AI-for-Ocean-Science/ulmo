# run h5 file through ulmo

from ulmo.models import io as model_io

pae = model_io.load_modis_l2(flavor='std', local=False)

pae.eval_data_file('LLC_modis2012_test_preproc.h5', 'valid', 'LLC_modis2012_test_preproc_log_prob.h5')