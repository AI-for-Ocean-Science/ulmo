import xarray as xr

fn = "https://opendap.jpl.nasa.gov/opendap/SeaSurfaceTopography/merged_alt/L4/cdr_grid/ssh_grids_v1812_1992100212.nc"

def print_metadata(data = fn):
	ds = xr.open_dataset(data)
	print(ds)
    
print_metadata()