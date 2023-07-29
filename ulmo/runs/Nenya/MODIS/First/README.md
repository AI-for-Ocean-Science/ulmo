# SimCLR for MODIS data set (version 1.0)

## Hyper-parmmeters in opt.json file
* lr_decay_epochs: epochs for the learning rate decay;
* weight_decay: coefficient of the Regularization;
* mean: mean used for the preprocess of the training samples;
* std: standard deviation for the preprocess of the training sampes;
* temp: temperature in the loss;
* cosine: flag of the cosine method for warm-up learning rate;
* syncBN: flag for the synchronization of the BN;
* warm: flag for warm-up learning rate;
* trial: id of the experiment;

## Running
Please put the modis .h5 dataset into the `./experiment/datasets/` directory and use the `train_modis.ipynb` to monitor the training process.

## Reference
This program is highly based on following repositories and papers. Actually we used most of the codes in 'SupContrast'(https://github.com/HobbitLong/SupContrast) and the we write our own 
dataloader and augmentation modules based on the codes in 'ssl-sky-surveys'(https://github.com/MustafaMustafa/ssl-sky-surveys).

