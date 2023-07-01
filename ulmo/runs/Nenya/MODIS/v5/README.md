# Dataset:

MODIS + 96% clear

# SSL model

Same as v4

## Copied v4 table to v5

```
cp MODIS_SSL_v4.parquet MODIS_Nenya_v5.parquet
```

# UMAP

Rerun of UMAP (pickle problems!!)

## Ran

```
python nenya_modis_v5.py --func_flag umap --debug --local
```

## conda list

packages in environment at /home/xavier/miniconda3/envs/os:

```
  Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                  2_kmp_llvm    conda-forge
abseil-cpp                20211102.0           h27087fc_1    conda-forge
aiofiles                  22.1.0                   pypi_0    pypi
aiosqlite                 0.18.0                   pypi_0    pypi
anyio                     3.6.2                    pypi_0    pypi
aom                       3.5.0                h27087fc_0    conda-forge
appdirs                   1.4.4              pyhd3eb1b0_0  
argon2-cffi               21.3.0                   pypi_0    pypi
argon2-cffi-bindings      21.2.0                   pypi_0    pypi
arrow                     1.2.3                    pypi_0    pypi
arrow-cpp                 8.0.0           py310h3098874_0  
astropy                   5.1             py310ha9d4c09_0  
astropy-healpix           0.7                      pypi_0    pypi
asttokens                 2.0.5              pyhd3eb1b0_0  
attrs                     22.2.0             pyh71513ae_0    conda-forge
aws-c-common              0.4.57               he6710b0_1  
aws-c-event-stream        0.1.6                h2531618_5  
aws-checksums             0.1.9                he6710b0_0  
aws-sdk-cpp               1.8.185              hce553d0_0  
babel                     2.12.1                   pypi_0    pypi
backcall                  0.2.0              pyhd3eb1b0_0  
beautifulsoup4            4.11.2                   pypi_0    pypi
blas                      1.0                         mkl  
bleach                    6.0.0                    pypi_0    pypi
blosc                     1.21.3               hafa529b_0    conda-forge
bokeh                     3.0.3           py310h06a4308_0  
boost-cpp                 1.70.0               ha2d47e9_1    conda-forge
boto3                     1.24.28         py310h06a4308_0  
botocore                  1.27.59         py310h06a4308_0  
bottleneck                1.3.5           py310ha9d4c09_0  
brotli                    1.0.9                h5eee18b_7  
brotli-bin                1.0.9                h5eee18b_7  
brotlipy                  0.7.0           py310h7f8727e_1002  
brunsli                   0.1                  h9c3ff4c_0    conda-forge
bzip2                     1.0.8                h7b6447c_0  
c-ares                    1.18.1               h7f98852_0    conda-forge
c-blosc2                  2.7.1                hf91038e_0    conda-forge
ca-certificates           2022.12.7            ha878542_0    conda-forge
cached-property           1.5.2                hd8ed1ab_1    conda-forge
cached_property           1.5.2              pyha770c72_1    conda-forge
cartopy                   0.21.1          py310h1176785_0  
certifi                   2022.12.7                pypi_0    pypi
cffi                      1.15.1          py310h5eee18b_3  
cfitsio                   4.2.0                hd9d235c_0    conda-forge
cftime                    1.6.2                    pypi_0    pypi
charls                    2.4.1                hcb278e6_0    conda-forge
charset-normalizer        2.0.4              pyhd3eb1b0_0  
click                     8.1.3           unix_pyhd8ed1ab_2    conda-forge
cloudpickle               2.2.1              pyhd8ed1ab_0    conda-forge
colorama                  0.4.6              pyhd8ed1ab_0    conda-forge
comm                      0.1.2                    pypi_0    pypi
contourpy                 1.0.5           py310hdb19cb5_0  
cryptography              39.0.1          py310h9ce1e76_0  
cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
cytoolz                   0.12.0          py310h5764c6d_1    conda-forge
dask-core                 2023.3.0           pyhd8ed1ab_0    conda-forge
dav1d                     1.0.0                h166bdaf_1    conda-forge
dbus                      1.13.18              hb2f20db_0  
debugpy                   1.6.6                    pypi_0    pypi
decorator                 5.1.1              pyhd3eb1b0_0  
defusedxml                0.7.1                    pypi_0    pypi
exceptiongroup            1.1.0              pyhd8ed1ab_0    conda-forge
executing                 0.8.3              pyhd3eb1b0_0  
expat                     2.4.9                h6a678d5_0  
extension-helpers         1.0.0                    pypi_0    pypi
fastjsonschema            2.16.3                   pypi_0    pypi
fontconfig                2.14.1               h52c9d5c_1  
fonttools                 4.39.0                   pypi_0    pypi
fqdn                      1.5.1                    pypi_0    pypi
freetype                  2.10.4               h0708190_1    conda-forge
fsspec                    2023.3.0           pyhd8ed1ab_1    conda-forge
geos                      3.8.0                he6710b0_0  
gflags                    2.2.2             he1b5a44_1004    conda-forge
giflib                    5.2.1                h36c2ea0_2    conda-forge
glib                      2.69.1               he621ea3_2  
glog                      0.6.0                h6f12383_0    conda-forge
grpc-cpp                  1.46.1               h33aed49_1  
gst-plugins-base          1.14.1               h6a678d5_1  
gstreamer                 1.14.1               h5eee18b_1  
h5netcdf                  1.1.0              pyhd8ed1ab_1    conda-forge
h5py                      3.8.0           nompi_py310ha66b2ad_101    conda-forge
hdf5                      1.14.0          nompi_h5231ba7_103    conda-forge
healpy                    1.16.2          py310h63f94f6_0    conda-forge
icu                       58.2                 he6710b0_3  
idna                      3.4             py310h06a4308_0  
imagecodecs               2023.1.23       py310ha3ed6a1_0    conda-forge
imageio                   2.26.0             pyh24c5eb1_0    conda-forge
importlib-metadata        6.0.0                    pypi_0    pypi
iniconfig                 2.0.0              pyhd8ed1ab_0    conda-forge
intel-openmp              2021.4.0          h06a4308_3561  
ipykernel                 6.21.3                   pypi_0    pypi
ipympl                    0.9.3                    pypi_0    pypi
ipython                   8.10.0          py310h06a4308_0  
ipython-genutils          0.2.0                    pypi_0    pypi
ipywidgets                8.0.6                    pypi_0    pypi
isoduration               20.11.0                  pypi_0    pypi
jedi                      0.18.1          py310h06a4308_1  
jinja2                    3.1.2           py310h06a4308_0  
jmespath                  0.10.0             pyhd3eb1b0_0  
joblib                    1.2.0              pyhd8ed1ab_0    conda-forge
jpeg                      9e                   h166bdaf_1    conda-forge
json5                     0.9.11                   pypi_0    pypi
jsonpointer               2.3                      pypi_0    pypi
jsonschema                4.17.3                   pypi_0    pypi
jupyter-client            8.0.3                    pypi_0    pypi
jupyter-core              5.2.0                    pypi_0    pypi
jupyter-events            0.6.3                    pypi_0    pypi
jupyter-server            2.4.0                    pypi_0    pypi
jupyter-server-fileid     0.8.0                    pypi_0    pypi
jupyter-server-terminals  0.4.4                    pypi_0    pypi
jupyter-server-ydoc       0.6.1                    pypi_0    pypi
jupyter-ydoc              0.2.2                    pypi_0    pypi
jupyterlab                3.6.1                    pypi_0    pypi
jupyterlab-pygments       0.2.2                    pypi_0    pypi
jupyterlab-server         2.20.0                   pypi_0    pypi
jupyterlab-vim            0.15.1                   pypi_0    pypi
jupyterlab-widgets        3.0.7                    pypi_0    pypi
jxrlib                    1.1                  h7f98852_2    conda-forge
keyutils                  1.6.1                h166bdaf_0    conda-forge
kiwisolver                1.4.4           py310h6a678d5_0  
krb5                      1.19.3               h3790be6_0    conda-forge
lcms2                     2.15                 hfd0df8a_0    conda-forge
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      4.0.0                h27087fc_0    conda-forge
libaec                    1.0.6                hcb278e6_1    conda-forge
libavif                   0.11.1               h5cdd6b5_0    conda-forge
libblas                   3.9.0            12_linux64_mkl    conda-forge
libbrotlicommon           1.0.9                h5eee18b_7  
libbrotlidec              1.0.9                h5eee18b_7  
libbrotlienc              1.0.9                h5eee18b_7  
libcblas                  3.9.0            12_linux64_mkl    conda-forge
libclang                  10.0.1          default_hb85057a_2  
libcurl                   7.87.0               h91b91d3_0  
libdeflate                1.17                 h5eee18b_0  
libedit                   3.1.20191231         he28a2e2_2    conda-forge
libev                     4.33                 h516909a_1    conda-forge
libevent                  2.1.12               h8f2d780_0  
libffi                    3.4.2                h6a678d5_6  
libgcc                    7.2.0                h69d50b8_2    conda-forge
libgcc-ng                 12.2.0              h65d4601_19    conda-forge
libgfortran-ng            12.2.0              h69a702a_19    conda-forge
libgfortran5              12.2.0              h337968e_19    conda-forge
libllvm10                 10.0.1               hbcb73fb_5  
libllvm11                 11.1.0               h9e868ea_6  
libnghttp2                1.46.0               hce63b2e_0  
libpng                    1.6.39               h5eee18b_0  
libpq                     12.9                 h16c4e8d_3  
libprotobuf               3.20.3               he621ea3_0  
libssh2                   1.10.0               ha56f1ee_2    conda-forge
libstdcxx-ng              12.2.0              h46fd767_19    conda-forge
libthrift                 0.15.0               hcc01f38_0  
libtiff                   4.5.0                h6adf6a1_2    conda-forge
libuuid                   1.41.5               h5eee18b_0  
libwebp                   1.2.4                h11a3e52_1  
libwebp-base              1.2.4                h5eee18b_1  
libxcb                    1.15                 h7f8727e_0  
libxkbcommon              1.0.1                hfa300c1_0  
libxml2                   2.9.14               h74e7548_0  
libxslt                   1.1.35               h4e12654_0  
libzlib                   1.2.13               h166bdaf_4    conda-forge
libzopfli                 1.0.3                h9c3ff4c_0    conda-forge
llvm-openmp               15.0.7               h0cdce71_0    conda-forge
llvmlite                  0.39.1          py310he621ea3_0  
locket                    1.0.0              pyhd8ed1ab_0    conda-forge
lz4-c                     1.9.3                h9c3ff4c_1    conda-forge
markupsafe                2.1.1           py310h7f8727e_0  
matplotlib                3.7.0           py310h06a4308_0  
matplotlib-base           3.7.0           py310h1128e8f_0  
matplotlib-inline         0.1.6           py310h06a4308_0  
mistune                   2.0.5                    pypi_0    pypi
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0           py310h7f8727e_0  
mkl_fft                   1.3.1           py310hd6ae3a3_0  
mkl_random                1.2.2           py310h00e6091_0  
munkres                   1.1.4                      py_0  
nbclassic                 0.5.3                    pypi_0    pypi
nbclient                  0.7.2                    pypi_0    pypi
nbconvert                 7.2.9                    pypi_0    pypi
nbformat                  5.7.3                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.5.6                    pypi_0    pypi
networkx                  3.0                pyhd8ed1ab_0    conda-forge
nodejs                    0.1.1                    pypi_0    pypi
notebook                  6.5.3                    pypi_0    pypi
notebook-shim             0.2.2                    pypi_0    pypi
nspr                      4.33                 h295c915_0  
nss                       3.74                 h0370c37_0  
numba                     0.56.4          py310ha5257ce_0    conda-forge
numexpr                   2.8.4           py310h8879344_0  
numpy                     1.23.5          py310hd5efca6_0  
numpy-base                1.23.5          py310h8e6c178_0  
oceanpy                   0.0.dev14                 dev_0    <develop>
openjpeg                  2.5.0                hfec8fc6_2    conda-forge
openssl                   1.1.1t               h0b41bf4_0    conda-forge
optional-django           0.1.0                    pypi_0    pypi
orc                       1.7.4                h07ed6aa_0  
packaging                 22.0            py310h06a4308_0  
pandas                    1.5.3           py310h1128e8f_0  
pandocfilters             1.5.0                    pypi_0    pypi
parso                     0.8.3              pyhd3eb1b0_0  
partd                     1.3.0              pyhd8ed1ab_0    conda-forge
patsy                     0.5.3           py310h06a4308_0  
pcre                      8.45                 h295c915_0  
pexpect                   4.8.0              pyhd3eb1b0_3  
pickleshare               0.7.5           pyhd3eb1b0_1003  
pillow                    9.4.0           py310h6a678d5_0  
pip                       23.0.1          py310h06a4308_0  
platformdirs              3.1.0                    pypi_0    pypi
pluggy                    1.0.0              pyhd8ed1ab_5    conda-forge
ply                       3.11            py310h06a4308_0  
pooch                     1.4.0              pyhd3eb1b0_0  
proj                      8.2.1                ha227179_0  
prometheus-client         0.16.0                   pypi_0    pypi
prompt-toolkit            3.0.36          py310h06a4308_0  
psutil                    5.9.4                    pypi_0    pypi
ptyprocess                0.7.0              pyhd3eb1b0_2  
pure_eval                 0.2.2              pyhd3eb1b0_0  
pyarrow                   8.0.0           py310h468efa6_0  
pycparser                 2.21               pyhd3eb1b0_0  
pyerfa                    2.0.0           py310h7f8727e_0  
pygments                  2.11.2             pyhd3eb1b0_0  
pynndescent               0.5.8                    pypi_0    pypi
pyopenssl                 23.0.0          py310h06a4308_0  
pyparsing                 3.0.9              pyhd8ed1ab_0    conda-forge
pyproj                    3.4.1           py310h49a4818_0  
pyqt                      5.15.7          py310h6a678d5_1  
pyqt5-sip                 12.11.0                  pypi_0    pypi
pyrsistent                0.19.3                   pypi_0    pypi
pyshp                     2.3.1              pyhd8ed1ab_0    conda-forge
pysocks                   1.7.1           py310h06a4308_0  
pytest                    7.2.2              pyhd8ed1ab_0    conda-forge
pytest-runner             6.0.0              pyhd8ed1ab_0    conda-forge
python                    3.10.9               h7a1cb2a_2  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python-json-logger        2.0.7                    pypi_0    pypi
python_abi                3.10                    2_cp310    conda-forge
pytz                      2022.7          py310h06a4308_0  
pywavelets                1.4.1           py310h0a54255_0    conda-forge
pyyaml                    6.0             py310h5eee18b_1  
pyzmq                     25.0.0                   pypi_0    pypi
qt-main                   5.15.2               h327a75a_7  
qt-webengine              5.15.9               hd2b0992_4  
qtwebkit                  5.212                h4eab89a_4  
re2                       2022.04.01           h27087fc_0    conda-forge
readline                  8.2                  h5eee18b_0  
requests                  2.28.1          py310h06a4308_0  
rfc3339-validator         0.1.4                    pypi_0    pypi
rfc3986-validator         0.1.1                    pypi_0    pypi
s3transfer                0.6.0           py310h06a4308_0  
scikit-image              0.19.3          py310h769672d_2    conda-forge
scikit-learn              1.2.2           py310h209a8ca_0    conda-forge
scipy                     1.10.0          py310hd5efca6_1  
seaborn                   0.12.2          py310h06a4308_0  
send2trash                1.8.0                    pypi_0    pypi
setuptools                65.6.3          py310h06a4308_0  
shapely                   1.8.4           py310h81ba7c5_0  
sip                       6.6.2           py310h6a678d5_0  
six                       1.16.0             pyhd3eb1b0_1  
smart-open                6.3.0                    pypi_0    pypi
snappy                    1.1.9                hbd366e4_2    conda-forge
sniffio                   1.3.0                    pypi_0    pypi
soupsieve                 2.4                      pypi_0    pypi
sqlite                    3.40.1               h5082296_0  
stack_data                0.2.0              pyhd3eb1b0_0  
statsmodels               0.13.5          py310ha9d4c09_1  
tbb                       2021.7.0             hdb19cb5_0  
terminado                 0.17.1                   pypi_0    pypi
threadpoolctl             3.1.0              pyh8a188c0_0    conda-forge
tifffile                  2023.2.28          pyhd8ed1ab_0    conda-forge
timm                      0.3.2                    pypi_0    pypi
tinycss2                  1.2.1                    pypi_0    pypi
tk                        8.6.12               h1ccaba5_0  
toml                      0.10.2             pyhd3eb1b0_0  
tomli                     2.0.1              pyhd8ed1ab_0    conda-forge
toolz                     0.12.0             pyhd8ed1ab_0    conda-forge
torch                     1.13.1+cpu               pypi_0    pypi
torchaudio                0.13.1+cpu               pypi_0    pypi
torchvision               0.14.1+cpu               pypi_0    pypi
tornado                   6.2                      pypi_0    pypi
tqdm                      4.65.0                   pypi_0    pypi
traitlets                 5.7.1           py310h06a4308_0  
typing-extensions         4.5.0                    pypi_0    pypi
tzdata                    2022g                h04d1e81_0  
ulmo                      0.0.dev0                  dev_0    <develop>
umap-learn                0.5.3                    pypi_0    pypi
uri-template              1.2.0                    pypi_0    pypi
urllib3                   1.26.14         py310h06a4308_0  
utf8proc                  2.6.1                h27cfd23_0  
wcwidth                   0.2.5              pyhd3eb1b0_0  
webcolors                 1.12                     pypi_0    pypi
webencodings              0.5.1                    pypi_0    pypi
websocket-client          1.5.1                    pypi_0    pypi
wheel                     0.38.4          py310h06a4308_0  
widgetsnbextension        4.0.7                    pypi_0    pypi
xarray                    2022.11.0       py310h06a4308_0  
xyzservices               2022.9.0        py310h06a4308_1  
xz                        5.2.10               h5eee18b_1  
y-py                      0.5.9                    pypi_0    pypi
yaml                      0.2.5                h7b6447c_0  
ypy-websocket             0.8.2                    pypi_0    pypi
zfp                       1.0.0                h27087fc_3    conda-forge
zipp                      3.15.0                   pypi_0    pypi
zlib                      1.2.13               h166bdaf_4    conda-forge
zlib-ng                   2.0.6                h166bdaf_0    conda-forge
zstd                      1.5.2                ha4553b6_0  
```