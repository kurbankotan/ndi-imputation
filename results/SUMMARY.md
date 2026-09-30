# NDI benchmark — summary

```
## Mean z-RMSE by setting (lower is better)
setting       MAR-30%  MCAR-10%  MCAR-30%  MCAR-50%  MNAR-30%
method                                                       
Copula-NDI-F    0.833     0.759     0.785     0.830     1.031
EM-FA           0.782     0.708     0.746     0.799     0.934
EM-PPCA         0.863     0.750     0.788     0.832     0.977
GAIN            0.860     0.809     0.829     0.987     0.999
GaussCM         0.806     0.680     0.737     0.800     1.026
HyperImpute     0.843       NaN     0.911       NaN     0.942
KNN             0.836     0.787     0.865     0.942     1.119
MICE            0.816     0.677     0.804     0.877     0.976
MIWAE           0.852       NaN     0.857       NaN     1.046
Mean            1.023     1.004     0.994     0.996     1.222
MissForest      0.731     0.653     0.705     0.790     0.915
NDI-0           1.054     1.014     1.027     1.053     1.363
NDI-C           0.864     0.797     0.821     0.875     1.073
NDI-F           0.814     0.732     0.765     0.813     1.024
NDI-F-noclip    0.823     0.739     0.774     0.821     1.029
NDI-F1          0.882     0.847     0.851     0.871     1.084
NDI-S           0.840     0.790     0.801     0.831     1.064
ReMasker        0.827       NaN     0.778       NaN     1.004
Sinkhorn        0.828       NaN     0.782       NaN     1.066
SoftImpute      0.808     0.708     0.754     0.816     1.038

## Average rank by setting (lower is better)
setting       MAR-30%  MCAR-10%  MCAR-30%  MCAR-50%  MNAR-30%
method                                                       
Copula-NDI-F    11.00      8.87      9.87      7.80      9.87
EM-FA            5.40      4.53      5.07      3.33      3.20
EM-PPCA         11.07      8.07     10.00      7.40      6.27
GAIN            12.47     12.47     14.47     13.40      8.47
GaussCM          7.47      2.87      4.33      3.80     10.20
HyperImpute      7.21       NaN      6.64       NaN      4.71
KNN             11.07      9.47     14.67     12.67     15.07
MICE             9.73      3.40     11.13     11.13      7.00
MIWAE           12.47       NaN     13.87       NaN     12.13
Mean            17.93     15.33     18.00     13.60     18.13
MissForest       2.80      2.33      2.60      4.00      2.73
NDI-0           19.13     15.27     18.67     15.07     18.60
NDI-C           12.80     10.93     13.47     10.40     13.33
NDI-F            8.60      6.60      7.40      5.07      9.20
NDI-F-noclip    10.43      7.97      9.30      6.90     10.73
NDI-F1          12.97     12.50     14.10      9.37     14.73
NDI-S           10.20     10.20     11.27      6.27     12.73
ReMasker        10.20       NaN      9.00       NaN      8.67
Sinkhorn         8.87       NaN      9.13       NaN     12.33
SoftImpute       7.33      5.20      6.13      5.80     10.87

## z-RMSE on the naturally-incomplete datasets (extra MCAR masking of observed cells)
method   Copula-NDI-F  EM-FA  EM-PPCA   GAIN  GaussCM  HyperImpute    KNN   MICE  MIWAE   Mean  MissForest  NDI-0  NDI-C  NDI-F  NDI-F-noclip  NDI-F1  NDI-S  ReMasker  Sinkhorn  SoftImpute
dataset                                                                                                                                                                                     
CKD             0.855  0.826    0.860  0.996    0.843        0.680  0.973  0.979  0.850  0.971       0.795  1.027  0.851  0.833         0.846   0.832  0.825     0.833     0.730       0.809
Heart           0.969  0.938    0.949  1.025    0.941        0.968  1.023  0.939  1.054  0.998       0.992  1.172  1.012  0.974         0.975   0.953  0.941     1.035     1.044       0.954
Mice            0.510  0.497    0.503  0.516    0.445        0.364  0.471  0.435  0.446  1.004       0.410  0.918  0.698  0.499         0.499   0.873  0.707     0.714     0.386       0.429

## Share of imputed values outside the observed range of the column (all settings)
method
KNN             0.0000
HyperImpute     0.0000
NDI-F           0.0000
Copula-NDI-F    0.0000
Mean            0.0006
MissForest      0.0041
ReMasker        0.0064
GAIN            0.0127
NDI-F1          0.0188
NDI-S           0.0223
SoftImpute      0.0251
Sinkhorn        0.0373
EM-FA           0.0409
NDI-0           0.0459
NDI-C           0.0466
EM-PPCA         0.0477
MICE            0.0526
GaussCM         0.0549
NDI-F-noclip    0.0641
MIWAE           0.0796

## Statistical tests

### Setting MAR-30%   (N = 15 datasets, k = 19 methods)
   Friedman chi2 = 115.73, p = 2.67e-16;  Nemenyi CD (alpha=0.05) = 7.227
   MissForest   average rank  2.60
   EM-FA        average rank  4.93  (not significantly worse than the best)
   SoftImpute   average rank  6.73  (not significantly worse than the best)
   GaussCM      average rank  6.93  (not significantly worse than the best)
   NDI-F        average rank  7.93  (not significantly worse than the best)
   Sinkhorn     average rank  8.27  (not significantly worse than the best)
   MICE         average rank  9.27  (not significantly worse than the best)
   NDI-S        average rank  9.53  (not significantly worse than the best)
   ReMasker     average rank  9.67  (not significantly worse than the best)
   NDI-F-noclip average rank  9.77  (not significantly worse than the best)
   EM-PPCA      average rank 10.33  *
   Copula-NDI-F average rank 10.33  *
   KNN          average rank 10.33  *
   GAIN         average rank 11.73  *
   MIWAE        average rank 11.87  *
   NDI-C        average rank 12.07  *
   NDI-F1       average rank 12.30  *
   Mean         average rank 17.13  *
   NDI-0        average rank 18.27  *
   * = significantly worse than the best-ranked method (Nemenyi, alpha = 0.05)

### Setting MCAR-10%   (N = 15 datasets, k = 16 methods)
   Friedman chi2 = 172.46, p = 7.85e-29;  Nemenyi CD (alpha=0.05) = 5.956
   MissForest   average rank  2.33
   GaussCM      average rank  2.87  (not significantly worse than the best)
   MICE         average rank  3.40  (not significantly worse than the best)
   EM-FA        average rank  4.53  (not significantly worse than the best)
   SoftImpute   average rank  5.20  (not significantly worse than the best)
   NDI-F        average rank  6.60  (not significantly worse than the best)
   NDI-F-noclip average rank  7.97  (not significantly worse than the best)
   EM-PPCA      average rank  8.07  (not significantly worse than the best)
   Copula-NDI-F average rank  8.87  *
   KNN          average rank  9.47  *
   NDI-S        average rank 10.20  *
   NDI-C        average rank 10.93  *
   GAIN         average rank 12.47  *
   NDI-F1       average rank 12.50  *
   NDI-0        average rank 15.27  *
   Mean         average rank 15.33  *
   * = significantly worse than the best-ranked method (Nemenyi, alpha = 0.05)

### Setting MCAR-30%   (N = 15 datasets, k = 19 methods)
   Friedman chi2 = 158.75, p = 1.46e-24;  Nemenyi CD (alpha=0.05) = 7.227
   MissForest   average rank  2.47
   GaussCM      average rank  3.87  (not significantly worse than the best)
   EM-FA        average rank  4.47  (not significantly worse than the best)
   SoftImpute   average rank  5.47  (not significantly worse than the best)
   NDI-F        average rank  6.73  (not significantly worse than the best)
   ReMasker     average rank  8.47  (not significantly worse than the best)
   Sinkhorn     average rank  8.47  (not significantly worse than the best)
   NDI-F-noclip average rank  8.63  (not significantly worse than the best)
   Copula-NDI-F average rank  9.20  (not significantly worse than the best)
   EM-PPCA      average rank  9.40  (not significantly worse than the best)
   MICE         average rank 10.40  *
   NDI-S        average rank 10.60  *
   NDI-C        average rank 12.73  *
   MIWAE        average rank 13.13  *
   NDI-F1       average rank 13.30  *
   GAIN         average rank 13.67  *
   KNN          average rank 13.93  *
   Mean         average rank 17.20  *
   NDI-0        average rank 17.87  *
   * = significantly worse than the best-ranked method (Nemenyi, alpha = 0.05)

### Setting MCAR-50%   (N = 15 datasets, k = 16 methods)
   Friedman chi2 = 145.05, p = 2.32e-23;  Nemenyi CD (alpha=0.05) = 5.956
   EM-FA        average rank  3.33
   GaussCM      average rank  3.80  (not significantly worse than the best)
   MissForest   average rank  4.00  (not significantly worse than the best)
   NDI-F        average rank  5.07  (not significantly worse than the best)
   SoftImpute   average rank  5.80  (not significantly worse than the best)
   NDI-S        average rank  6.27  (not significantly worse than the best)
   NDI-F-noclip average rank  6.90  (not significantly worse than the best)
   EM-PPCA      average rank  7.40  (not significantly worse than the best)
   Copula-NDI-F average rank  7.80  (not significantly worse than the best)
   NDI-F1       average rank  9.37  *
   NDI-C        average rank 10.40  *
   MICE         average rank 11.13  *
   KNN          average rank 12.67  *
   GAIN         average rank 13.40  *
   Mean         average rank 13.60  *
   NDI-0        average rank 15.07  *
   * = significantly worse than the best-ranked method (Nemenyi, alpha = 0.05)

### Setting MNAR-30%   (N = 15 datasets, k = 19 methods)
   Friedman chi2 = 150.49, p = 5.97e-23;  Nemenyi CD (alpha=0.05) = 7.227
   MissForest   average rank  2.33
   EM-FA        average rank  2.73  (not significantly worse than the best)
   EM-PPCA      average rank  5.60  (not significantly worse than the best)
   MICE         average rank  6.27  (not significantly worse than the best)
   GAIN         average rank  7.67  (not significantly worse than the best)
   ReMasker     average rank  8.00  (not significantly worse than the best)
   NDI-F        average rank  8.40  (not significantly worse than the best)
   Copula-NDI-F average rank  9.20  (not significantly worse than the best)
   GaussCM      average rank  9.40  (not significantly worse than the best)
   NDI-F-noclip average rank  9.93  *
   SoftImpute   average rank 10.07  *
   MIWAE        average rank 11.33  *
   Sinkhorn     average rank 11.47  *
   NDI-S        average rank 11.93  *
   NDI-C        average rank 12.53  *
   NDI-F1       average rank 13.93  *
   KNN          average rank 14.20  *
   Mean         average rank 17.27  *
   NDI-0        average rank 17.73  *
   * = significantly worse than the best-ranked method (Nemenyi, alpha = 0.05)

### All settings pooled (blocks = dataset x setting; methods with a complete set of blocks)   (N = 75 datasets, k = 16 methods)
   Friedman chi2 = 637.46, p = 3.87e-126;  Nemenyi CD (alpha=0.05) = 2.664
   MissForest   average rank  2.61
   EM-FA        average rank  3.59  (not significantly worse than the best)
   GaussCM      average rank  4.72  (not significantly worse than the best)
   SoftImpute   average rank  5.96  *
   NDI-F        average rank  6.21  *
   MICE         average rank  7.35  *
   EM-PPCA      average rank  7.47  *
   NDI-F-noclip average rank  7.87  *
   Copula-NDI-F average rank  8.32  *
   NDI-S        average rank  8.67  *
   NDI-C        average rank 10.53  *
   GAIN         average rank 10.83  *
   KNN          average rank 10.92  *
   NDI-F1       average rank 11.11  *
   Mean         average rank 14.60  *
   NDI-0        average rank 15.25  *
   * = significantly worse than the best-ranked method (Nemenyi, alpha = 0.05)

### Wilcoxon signed-rank, NDI-F vs each method (blocks = dataset x setting, Holm-corrected)
   vs NDI-0        blocks= 75  NDI-F better in  75, worse in   0;  mean diff (z-RMSE) = -0.2726;  p = 5.28e-14, Holm p = 1.00e-12
   vs NDI-F-noclip blocks= 75  NDI-F better in  74, worse in   1;  mean diff (z-RMSE) = -0.0077;  p = 5.50e-14, Holm p = 9.90e-13
   vs Mean         blocks= 75  NDI-F better in  73, worse in   2;  mean diff (z-RMSE) = -0.2182;  p = 6.46e-14, Holm p = 1.10e-12
   vs NDI-C        blocks= 75  NDI-F better in  66, worse in   9;  mean diff (z-RMSE) = -0.0565;  p = 4.11e-12, Holm p = 6.57e-11
   vs MissForest   blocks= 75  NDI-F better in   9, worse in  66;  mean diff (z-RMSE) = +0.0707;  p = 1.34e-11, Holm p = 2.01e-10
   vs NDI-F1       blocks= 75  NDI-F better in  67, worse in   8;  mean diff (z-RMSE) = -0.0775;  p = 1.06e-10, Holm p = 1.48e-09
   vs EM-FA        blocks= 75  NDI-F better in  11, worse in  64;  mean diff (z-RMSE) = +0.0357;  p = 3.66e-10, Holm p = 4.75e-09
   vs GAIN         blocks= 75  NDI-F better in  62, worse in  13;  mean diff (z-RMSE) = -0.0673;  p = 4.10e-08, Holm p = 4.92e-07
   vs KNN          blocks= 75  NDI-F better in  61, worse in  14;  mean diff (z-RMSE) = -0.0802;  p = 9.36e-08, Holm p = 1.03e-06
   vs Copula-NDI-F blocks= 75  NDI-F better in  55, worse in  20;  mean diff (z-RMSE) = -0.0177;  p = 7.11e-07, Holm p = 7.11e-06
   vs NDI-S        blocks= 75  NDI-F better in  53, worse in  22;  mean diff (z-RMSE) = -0.0356;  p = 4.69e-06, Holm p = 4.22e-05
   vs GaussCM      blocks= 75  NDI-F better in  22, worse in  53;  mean diff (z-RMSE) = +0.0197;  p = 2.03e-05, Holm p = 1.63e-04
   vs MIWAE        blocks= 45  NDI-F better in  31, worse in  14;  mean diff (z-RMSE) = -0.0508;  p = 5.11e-03, Holm p = 3.57e-02
   vs HyperImpute  blocks= 42  NDI-F better in  10, worse in  32;  mean diff (z-RMSE) = -0.0274;  p = 1.09e-02, Holm p = 6.55e-02
   vs SoftImpute   blocks= 75  NDI-F better in  26, worse in  49;  mean diff (z-RMSE) = +0.0045;  p = 2.45e-02, Holm p = 1.22e-01
   vs Sinkhorn     blocks= 45  NDI-F better in  26, worse in  19;  mean diff (z-RMSE) = -0.0241;  p = 5.02e-02, Holm p = 2.01e-01
   vs EM-PPCA      blocks= 75  NDI-F better in  42, worse in  33;  mean diff (z-RMSE) = -0.0123;  p = 1.27e-01, Holm p = 3.81e-01
   vs MICE         blocks= 75  NDI-F better in  38, worse in  37;  mean diff (z-RMSE) = -0.0005;  p = 8.53e-01, Holm p = 1.00e+00
   vs ReMasker     blocks= 45  NDI-F better in  23, worse in  22;  mean diff (z-RMSE) = -0.0020;  p = 9.64e-01, Holm p = 9.64e-01

### Pooled over MCAR-30%, MAR-30%, MNAR-30% (blocks = dataset x setting; all methods incl. deep baselines)   (N = 45 datasets, k = 19 methods)
   Friedman chi2 = 379.13, p = 2.03e-69;  Nemenyi CD (alpha=0.05) = 4.172
   MissForest   average rank  2.47
   EM-FA        average rank  4.04  (not significantly worse than the best)
   GaussCM      average rank  6.73  *
   SoftImpute   average rank  7.42  *
   NDI-F        average rank  7.69  *
   EM-PPCA      average rank  8.44  *
   MICE         average rank  8.64  *
   ReMasker     average rank  8.71  *
   Sinkhorn     average rank  9.40  *
   NDI-F-noclip average rank  9.44  *
   Copula-NDI-F average rank  9.58  *
   NDI-S        average rank 10.69  *
   GAIN         average rank 11.02  *
   MIWAE        average rank 12.11  *
   NDI-C        average rank 12.44  *
   KNN          average rank 12.82  *
   NDI-F1       average rank 13.18  *
   Mean         average rank 17.20  *
   NDI-0        average rank 17.96  *
   * = significantly worse than the best-ranked method (Nemenyi, alpha = 0.05)

### Wilcoxon signed-rank, NDI-F vs each method (blocks = dataset x 30 % settings, Holm-corrected)
   vs NDI-0        blocks= 45  NDI-F better in  45, worse in   0;  mean diff (z-RMSE) = -0.2802;  p = 5.68e-14, Holm p = 1.08e-12
   vs NDI-F-noclip blocks= 45  NDI-F better in  45, worse in   0;  mean diff (z-RMSE) = -0.0077;  p = 5.68e-14, Holm p = 1.02e-12
   vs Mean         blocks= 45  NDI-F better in  44, worse in   1;  mean diff (z-RMSE) = -0.2122;  p = 1.71e-13, Holm p = 2.90e-12
   vs MissForest   blocks= 45  NDI-F better in   2, worse in  43;  mean diff (z-RMSE) = +0.0841;  p = 3.98e-12, Holm p = 6.37e-11
   vs NDI-C        blocks= 45  NDI-F better in  39, worse in   6;  mean diff (z-RMSE) = -0.0519;  p = 8.42e-10, Holm p = 1.26e-08
   vs EM-FA        blocks= 45  NDI-F better in   7, worse in  38;  mean diff (z-RMSE) = +0.0471;  p = 9.97e-09, Holm p = 1.40e-07
   vs NDI-F1       blocks= 45  NDI-F better in  40, worse in   5;  mean diff (z-RMSE) = -0.0714;  p = 1.15e-07, Holm p = 1.50e-06
   vs KNN          blocks= 45  NDI-F better in  36, worse in   9;  mean diff (z-RMSE) = -0.0722;  p = 5.14e-05, Holm p = 6.16e-04
   vs NDI-S        blocks= 45  NDI-F better in  33, worse in  12;  mean diff (z-RMSE) = -0.0341;  p = 2.01e-04, Holm p = 2.22e-03
   vs Copula-NDI-F blocks= 45  NDI-F better in  31, worse in  14;  mean diff (z-RMSE) = -0.0152;  p = 3.02e-04, Holm p = 3.02e-03
   vs GAIN         blocks= 45  NDI-F better in  32, worse in  13;  mean diff (z-RMSE) = -0.0284;  p = 4.56e-03, Holm p = 4.10e-02
   vs MIWAE        blocks= 45  NDI-F better in  31, worse in  14;  mean diff (z-RMSE) = -0.0508;  p = 5.11e-03, Holm p = 4.08e-02
   vs HyperImpute  blocks= 42  NDI-F better in  10, worse in  32;  mean diff (z-RMSE) = -0.0274;  p = 1.09e-02, Holm p = 7.64e-02
   vs GaussCM      blocks= 45  NDI-F better in  16, worse in  29;  mean diff (z-RMSE) = +0.0114;  p = 3.01e-02, Holm p = 1.81e-01
   vs Sinkhorn     blocks= 45  NDI-F better in  26, worse in  19;  mean diff (z-RMSE) = -0.0241;  p = 5.02e-02, Holm p = 2.51e-01
   vs SoftImpute   blocks= 45  NDI-F better in  17, worse in  28;  mean diff (z-RMSE) = +0.0008;  p = 2.21e-01, Holm p = 8.84e-01
   vs MICE         blocks= 45  NDI-F better in  21, worse in  24;  mean diff (z-RMSE) = +0.0023;  p = 8.93e-01, Holm p = 1.00e+00
   vs EM-PPCA      blocks= 45  NDI-F better in  21, worse in  24;  mean diff (z-RMSE) = -0.0083;  p = 9.02e-01, Holm p = 1.00e+00
   vs ReMasker     blocks= 45  NDI-F better in  23, worse in  22;  mean diff (z-RMSE) = -0.0020;  p = 9.64e-01, Holm p = 9.64e-01

## Method failures
method       error                                                                     
HyperImpute  XGBoostError: value 0 for Parameter num_class should be greater equal to 1    3

## Downstream accuracy, mean over datasets, seeds and the 4 classifiers
setting       MAR-30%  MCAR-30%  MNAR-30%  complete data
method                                                  
COMPLETE          NaN       NaN       NaN         0.8434
Copula-NDI-F   0.7929    0.7897    0.7850            NaN
EM-FA          0.7970    0.7933    0.7971            NaN
EM-PPCA        0.7916    0.7878    0.7932            NaN
GAIN           0.7901    0.7834    0.7915            NaN
GaussCM        0.7920    0.7930    0.7866            NaN
HyperImpute    0.8071    0.7928    0.8032            NaN
KNN            0.7943    0.7828    0.7798            NaN
MICE           0.7954    0.7920    0.7959            NaN
MIWAE          0.8034    0.7924    0.7950            NaN
Mean           0.7829    0.7733    0.7812            NaN
MissForest     0.8052    0.8018    0.8023            NaN
NDI-0          0.7808    0.7688    0.7746            NaN
NDI-C          0.7894    0.7830    0.7841            NaN
NDI-F          0.7911    0.7892    0.7873            NaN
NDI-F-noclip   0.7908    0.7894    0.7862            NaN
NDI-F1         0.7878    0.7819    0.7835            NaN
NDI-S          0.7902    0.7848    0.7861            NaN
ReMasker       0.8049    0.7977    0.8046            NaN
Sinkhorn       0.8068    0.7973    0.8022            NaN
SoftImpute     0.7924    0.7915    0.7915            NaN

## Complete-data reference accuracy per dataset (mean over classifiers)
dataset
MagicTelescope      0.8246
banknote            0.9931
diabetes            0.7556
ecoli               0.8774
glass               0.7012
ionosphere          0.9017
iris                0.9544
qsar-biodeg         0.8691
segment             0.9464
spambase            0.9158
vehicle             0.7566
wdbc                0.9701
wine                0.9771
wine-quality-red    0.6201
yeast               0.5877

## Downstream accuracy on CKD (its own missing values imputed; 5-fold CV, mean over seeds)
classifier       KNN      LR      RF     SVC    mean
method                                              
Copula-NDI-F  0.9708  0.9908  0.9950  0.9875  0.9860
EM-FA         0.9658  0.9925  0.9925  0.9900  0.9852
EM-PPCA       0.9642  0.9917  0.9950  0.9875  0.9846
GAIN          0.9625  0.9842  0.9900  0.9850  0.9804
GaussCM       0.9658  0.9925  0.9917  0.9875  0.9844
HyperImpute   0.9692  0.9925  0.9967  0.9917  0.9875
KNN           0.9625  0.9842  0.9758  0.9817  0.9760
MICE          0.9658  0.9925  0.9950  0.9883  0.9854
MIWAE         0.9692  0.9883  0.9942  0.9892  0.9852
Mean          0.9692  0.9942  0.9925  0.9958  0.9879
MissForest    0.9700  0.9900  0.9933  0.9892  0.9856
NDI-0         0.9683  0.9900  0.9892  0.9917  0.9848
NDI-C         0.9675  0.9900  0.9967  0.9875  0.9854
NDI-F         0.9650  0.9925  0.9942  0.9875  0.9848
NDI-F-noclip  0.9650  0.9925  0.9942  0.9875  0.9848
NDI-F1        0.9675  0.9917  0.9950  0.9875  0.9854
NDI-S         0.9675  0.9892  0.9975  0.9875  0.9854
ReMasker      0.9733  0.9892  0.9950  0.9900  0.9869
Sinkhorn      0.9700  0.9917  0.9967  0.9892  0.9869
SoftImpute    0.9667  0.9917  0.9933  0.9875  0.9848

## Downstream accuracy on Heart (its own missing values imputed; 5-fold CV, mean over seeds)
classifier       KNN      LR      RF     SVC    mean
method                                              
Copula-NDI-F  0.8245  0.8444  0.7936  0.8192  0.8204
EM-FA         0.8245  0.8444  0.7936  0.8192  0.8204
EM-PPCA       0.8245  0.8444  0.7936  0.8192  0.8204
GAIN          0.8245  0.8444  0.7947  0.8192  0.8207
GaussCM       0.8245  0.8444  0.7936  0.8192  0.8204
HyperImpute   0.8245  0.8444  0.7936  0.8192  0.8204
KNN           0.8245  0.8444  0.7936  0.8192  0.8204
MICE          0.8245  0.8444  0.7936  0.8192  0.8204
MIWAE         0.8245  0.8444  0.7947  0.8192  0.8207
Mean          0.8245  0.8444  0.7936  0.8192  0.8204
MissForest    0.8245  0.8444  0.7936  0.8192  0.8204
NDI-0         0.8245  0.8444  0.7936  0.8192  0.8204
NDI-C         0.8245  0.8444  0.7936  0.8192  0.8204
NDI-F         0.8245  0.8444  0.7936  0.8192  0.8204
NDI-F-noclip  0.8245  0.8444  0.7936  0.8192  0.8204
NDI-F1        0.8245  0.8444  0.7936  0.8192  0.8204
NDI-S         0.8245  0.8444  0.7936  0.8192  0.8204
ReMasker      0.8245  0.8444  0.7947  0.8192  0.8207
Sinkhorn      0.8245  0.8444  0.7936  0.8192  0.8204
SoftImpute    0.8245  0.8444  0.7936  0.8192  0.8204

## Downstream accuracy on Mice (its own missing values imputed; 5-fold CV, mean over seeds)
classifier       KNN      LR      RF     SVC    mean
method                                              
Copula-NDI-F  0.9728  0.9898  0.9920  0.9969  0.9879
EM-FA         0.9741  0.9880  0.9935  0.9969  0.9881
EM-PPCA       0.9728  0.9895  0.9929  0.9969  0.9880
GAIN          0.9741  0.9895  0.9944  0.9969  0.9887
GaussCM       0.9747  0.9907  0.9923  0.9969  0.9886
HyperImpute   0.9747  0.9895  0.9938  0.9969  0.9887
KNN           0.9744  0.9901  0.9941  0.9969  0.9889
MICE          0.9750  0.9898  0.9948  0.9969  0.9891
MIWAE         0.9738  0.9895  0.9932  0.9969  0.9884
Mean          0.9781  0.9901  0.9938  0.9966  0.9896
MissForest    0.9738  0.9889  0.9938  0.9960  0.9881
NDI-0         0.9790  0.9901  0.9941  0.9966  0.9900
NDI-C         0.9756  0.9898  0.9932  0.9966  0.9888
NDI-F         0.9738  0.9889  0.9926  0.9969  0.9880
NDI-F-noclip  0.9738  0.9886  0.9926  0.9969  0.9880
NDI-F1        0.9781  0.9904  0.9929  0.9966  0.9895
NDI-S         0.9762  0.9898  0.9935  0.9966  0.9890
ReMasker      0.9753  0.9904  0.9932  0.9966  0.9889
Sinkhorn      0.9744  0.9901  0.9941  0.9963  0.9887
SoftImpute    0.9738  0.9898  0.9929  0.9969  0.9884

## MissForest warm start (own loop, 20 trees, Stekhoven stopping rule, max 8 iterations, MCAR 30 %): mean over datasets and seeds
       rmse_init  rmse_after_1_iter  rmse_final  iterations  iters_to_within_1pct  time_s
init                                                                                     
NDI-F     0.7651             0.7047      0.7037      4.3111                1.7556  6.8470
mean      0.9942             0.7165      0.7049      4.8000                2.0444  7.7489

## Wall-clock seconds on synthetic data (MCAR 30 %); blank = skipped as too costly
p                 10                    50              
n             1000   10000  100000  1000   10000  100000
method                                                  
Copula-NDI-F    0.02   0.05   0.43    0.05   0.30   3.85
EM-FA           0.12   1.37   3.84    0.06   0.90   5.56
EM-PPCA         0.07   0.54   2.87    0.04   0.43   3.60
GAIN           10.20  10.57  10.41   10.43  10.61  10.84
GaussCM         0.04   0.08   0.26    0.18   1.37  15.29
HyperImpute    10.52    NaN    NaN   21.79    NaN    NaN
KNN             0.17    NaN    NaN    0.24    NaN    NaN
MICE            0.26   0.51    NaN    2.68  15.88    NaN
MIWAE           9.56    NaN    NaN    9.82    NaN    NaN
Mean            0.00   0.00   0.03    0.00   0.01   0.09
MissForest      2.18    NaN    NaN   14.50    NaN    NaN
NDI-0           0.00   0.01   0.07    0.00   0.03   0.26
NDI-C           0.00   0.01   0.11    0.01   0.16   1.74
NDI-F           0.00   0.03   0.35    0.02   0.18   1.83
NDI-F-noclip    0.00   0.02   0.36    0.01   0.18   1.72
NDI-F1          0.00   0.02   0.20    0.02   0.16   1.67
NDI-S           0.00   0.01   0.14    0.02   0.17   1.77
ReMasker      191.77    NaN    NaN  193.95    NaN    NaN
Sinkhorn      171.70    NaN    NaN  203.13    NaN    NaN
SoftImpute      0.07   0.57   8.50    1.34   2.63  47.19
```
