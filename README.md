Code Description (Example: Diurnal Tide Prediction)
1.	Pre-trained Model
o	Dseq_diur7_0.009571495.pth: Trained model for diurnal internal tide prediction.
2.	Data Preparation (MATLAB)
o	Training set extraction:
	Datareduction and K1_train: Key scripts for extracting grid points from CORA dataset (used for training). Example shown for one fold in k-fold cross-validation.
o	Valid set: K1_valid
o	Test set processing example: Informer_dis_scs, p_energy_dis
o	Mooring data processing: Informermooringdata
o	Kinetic energy error comparison: YY_KE_20N
3.	Deep Learning Prediction (Python)
o	Model training:
	Dseq_diur: Training code (includes validation set handling)
o	Harmonic analysis prediction: HHHHA
o	Mooring data prediction: Mooringdata_diur_dseq
o	Profile prediction (u-component at 20°N, later integrated for kinetic energy): Sci_u_region_Dseq_20N
o	Full South China Sea diurnal tide prediction: sci_diur_seq_scs_dep0

The deep learning framework for this research was adapted from the work published at:
https://blog.csdn.net/java1314777/article/details/134864319
