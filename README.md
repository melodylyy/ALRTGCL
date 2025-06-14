# ALRTGCL
MiRNA-drug sensitivity prediction 

The programs is supported by Python 3.9 . 

## Required Packages
numpy == 1.19.2
tensorflow== 1.15.0
matplotlib == 3.3.4
scipy == 1.5.2
scikit-learn == 1.5.1
# Input
* miRNA-drug sensitivity association network
* miRNA-miRNA association network
* drug-drug associations network
# Method
VCMDS calculates Gaussian Interaction Profile (GIP) kernel similarities for miRNAs and drugs, integrating these metrics into their respective similarity networks. We variational graph autoencoders and collaborative matrix factorization CMF to extract features. Finally, predicted scores are obtained using a fully connected network.

