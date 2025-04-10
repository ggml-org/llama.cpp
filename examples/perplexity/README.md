# Perplexity

The `perplexity` example can be used to calculate the so-called perplexity value of a language model over a given text corpus.
Perplexity measures how well the model can predict the next token with lower values being better.
Note that perplexity is **not** directly comparable between models, especially if they use different tokenizers.
Also note that finetunes typically result in a higher perplexity value even though the human-rated quality of outputs increases.

Within llama.cpp the perplexity of base models is used primarily to judge the quality loss from e.g. quantized models vs. FP16.
The convention among contributors is to use the Wikitext-2 test set for testing unless noted otherwise (can be obtained with `scripts/get-wikitext-2.sh`).
When numbers are listed all command line arguments and compilation options are left at their defaults unless noted otherwise.
llama.cpp numbers are **not** directly comparable to those of other projects because the exact values depend strongly on the implementation details.

By default only the mean perplexity value and the corresponding uncertainty is calculated.
The uncertainty is determined empirically by assuming a Gaussian distribution of the "correct" logits per and then applying error propagation.

More statistics can be obtained by recording the logits from the FP16 version of a model.
To do this, supply `perplexity` with `--kl-divergence-base path/to/logit/binary/file.kld`.
The program will then record all logits and save them to the provided path in binary format.
**The logit file will be very large, 11 GiB for LLaMA 2 or 37 GiB for LLaMA 3 when using the Wikitext-2 test set.**
Once you have the file, supply `perplexity` with the quantized model, the logits file via `--kl-divergence-base`,
and finally the `--kl-divergence` argument to indicate that the program should calculate the so-called Kullback-Leibler divergence.
This is a measure of how similar the FP16 and the quantized logit distributions are with a value of 0 indicating that the distribution are the same.
The uncertainty on the mean KL divergence is calculated by assuming the KL divergence per token follows a Gaussian distribution.

In addition to the KL divergence the following statistics are calculated with `--kl-divergence`:

* Ratio of mean FP16 PPL and quantized PPL. Uncertainty is estimated on logits, then propagated. The logarithm of this metric is also calculated and printed, it is 0 if the logit distributions are the same.
* Difference of mean FP16 PPL and quantized PPL. Uncertainty is estimated on logits, then propagated.
* Mean change in "correct" token probability. Positive values mean the model gets better at prediction, negative values mean it gets worse.
* Pearson correlation coefficient of the "correct" token probabilites between models.
* Percentiles of change in "correct" token probability. Positive values mean the model gets better at prediction, negative values mean it gets worse. Can be used to judge noise vs. quality loss from quantization. If the percentiles are symmetric then the quantization is essentially just adding noise. If the negative values are significantly larger than the positive values then this indicates that the model is actually becoming worse from the quantization.
* The root mean square of the change in token probabilities. If you were to assume that the quantization simply causes Gaussian noise on the token probabilities then this would be the standard deviation of said noise. The uncertainty on the value is calculated that the change in token probabilities follows a Gaussian distribution. Related discussion: https://github.com/ggerganov/llama.cpp/discussions/2875 .
* Same top p: Percentage of how often the token was assigned the highest probabilites by both models. The uncertainty is calculated from the Gaussian approximation of the binomial distribution.

## LLaMA 3 8b Scoreboard

| Revision | f364eb6f           |
|:---------|:-------------------|
| Backend  | CUDA               |
| CPU      | AMD Epyc 7742      |
| GPU      | 1x NVIDIA RTX 4090 |

Results were generated using the CUDA backend and are sorted by Kullback-Leibler divergence relative to FP16.
The "WT" importance matrices were created using varying numbers of Wikitext tokens and can be found [here](https://huggingface.co/JohannesGaessler/llama.cpp_importance_matrices/blob/main/imatrix-llama_3-8b-f16-2.7m_tokens.dat).
Note: the FP16 logits used for the calculation of all metrics other than perplexity are stored in a binary file between runs.
In order to save space this file does **not** contain the exact same FP32 logits but instead casts them to 16 bit unsigned integers (with some scaling).
So the "f16" results are to be understood as the difference resulting only from this downcast.

| Quantization | imatrix | Model size [GiB] | PPL                    | ΔPPL                   | KLD                   | Mean Δp           | RMS Δp           |
|--------------|---------|------------------|------------------------|------------------------|-----------------------|-------------------|------------------|
| f16          | None    |            14.97 | 6.233160 ±   0.037828  | 0.001524 ±   0.000755  | 0.000551 ±   0.000002 |  0.001 ± 0.002 %  | 0.787 ± 0.004 %  |
| q8_0         | None    |             7.96 | 6.234284 ±   0.037878  | 0.002650 ±   0.001006  | 0.001355 ±   0.000006 | -0.019 ± 0.003 %  | 1.198 ± 0.007 %  |
| q6_K         | None    |             6.14 | 6.253382 ±   0.038078  | 0.021748 ±   0.001852  | 0.005452 ±   0.000035 | -0.007 ± 0.006 %  | 2.295 ± 0.019 %  |
| q5_K_M       | None    |             5.33 | 6.288607 ±   0.038338  | 0.056974 ±   0.002598  | 0.010762 ±   0.000079 | -0.114 ± 0.008 %  | 3.160 ± 0.031 %  |
| q5_K_S       | None    |             5.21 | 6.336598 ±   0.038755  | 0.104964 ±   0.003331  | 0.016595 ±   0.000122 | -0.223 ± 0.010 %  | 3.918 ± 0.036 %  |
| q5_1         | None    |             5.65 | 6.337857 ±   0.038677  | 0.106223 ±   0.003476  | 0.018045 ±   0.000139 | -0.287 ± 0.011 %  | 4.123 ± 0.039 %  |
| q5_0         | None    |             5.21 | 6.363224 ±   0.038861  | 0.131591 ±   0.003894  | 0.022239 ±   0.000166 | -0.416 ± 0.012 %  | 4.634 ± 0.043 %  |
| q4_K_M       | WT 10m  |             4.58 | 6.382937 ±   0.039055  | 0.151303 ±   0.004429  | 0.028152 ±   0.000240 | -0.389 ± 0.014 %  | 5.251 ± 0.049 %  |
| q4_K_M       | None    |             4.58 | 6.407115 ±   0.039119  | 0.175482 ±   0.004620  | 0.031273 ±   0.000238 | -0.596 ± 0.014 %  | 5.519 ± 0.050 %  |
| q4_K_S       | WT 10m  |             4.37 | 6.409697 ±   0.039189  | 0.178064 ±   0.004744  | 0.031951 ±   0.000259 | -0.531 ± 0.015 %  | 5.645 ± 0.051 %  |
| iq4_NL       | WT 10m  |             4.35 | 6.455593 ±   0.039630  | 0.223959 ±   0.005201  | 0.035742 ±   0.000288 | -0.590 ± 0.016 %  | 5.998 ± 0.054 %  |
| iq4_XS       | WT 10m  |             4.14 | 6.459705 ±   0.039595  | 0.228071 ±   0.005207  | 0.036334 ±   0.000284 | -0.668 ± 0.016 %  | 6.044 ± 0.054 %  |
| q4_K_S       | None    |             4.37 | 6.500529 ±   0.039778  | 0.268895 ±   0.005638  | 0.043136 ±   0.000314 | -0.927 ± 0.017 %  | 6.562 ± 0.055 %  |
| q4_1         | None    |             4.78 | 6.682737 ±   0.041285  | 0.451103 ±   0.008030  | 0.071683 ±   0.000505 | -0.927 ± 0.017 %  | 8.512 ± 0.063 %  |
| q4_0         | None    |             4.34 | 6.700147 ±   0.041226  | 0.468514 ±   0.007951  | 0.071940 ±   0.000491 | -1.588 ± 0.022 %  | 8.434 ± 0.061 %  |
| q3_K_L       | WT 10m  |             4.03 | 6.671223 ±   0.041427  | 0.439590 ±   0.008154  | 0.073077 ±   0.000529 | -0.940 ± 0.023 %  | 8.662 ± 0.064 %  |
| q3_K_M       | WT 10m  |             3.74 | 6.734255 ±   0.041838  | 0.502622 ±   0.008901  | 0.084358 ±   0.000588 | -1.198 ± 0.024 %  | 9.292 ± 0.065 %  |
| q3_K_L       | None    |             4.03 | 6.787876 ±   0.042104  | 0.556242 ±   0.009171  | 0.087176 ±   0.000614 | -1.532 ± 0.025 %  | 9.432 ± 0.067 %  |
| q3_K_M       | None    |             3.74 | 6.888498 ±   0.042669  | 0.656864 ±   0.010071  | 0.101913 ±   0.000677 | -1.990 ± 0.026 %  | 10.203 ± 0.068 % |
| iq3_M        | WT 10m  |             3.53 | 6.898327 ±   0.041643  | 0.666694 ±   0.009449  | 0.102534 ±   0.000663 | -3.178 ± 0.026 %  | 10.513 ± 0.066 % |
| iq3_S        | WT 10m  |             3.42 | 6.965501 ±   0.042406  | 0.733867 ±   0.010245  | 0.111278 ±   0.000710 | -3.066 ± 0.027 %  | 10.845 ± 0.068 % |
| iq3_XS       | WT 10m  |             3.28 | 7.163043 ±   0.043772  | 0.931409 ±   0.012084  | 0.138693 ±   0.000857 | -3.667 ± 0.031 %  | 12.148 ± 0.070 % |
| iq3_XXS      | WT 10m  |             3.05 | 7.458436 ±   0.046404  | 1.226803 ±   0.015234  | 0.183625 ±   0.001042 | -3.918 ± 0.035 %  | 13.836 ± 0.074 % |
| q3_K_S       | WT 10m  |             3.41 | 7.602878 ±   0.046848  | 1.371244 ±   0.015688  | 0.199821 ±   0.001008 | -5.046 ± 0.037 %  | 14.980 ± 0.070 % |
| q3_K_S       | None    |             3.41 | 7.863786 ±   0.048885  | 1.632152 ±   0.017733  | 0.228217 ±   0.001079 | -5.604 ± 0.038 %  | 15.541 ± 0.070 % |
| iq2_M        | WT 10m  |             2.74 | 8.600799 ±   0.055124  | 2.369166 ±   0.025244  | 0.325989 ±   0.00160  | -6.463 ± 0.046 %  | 18.519 ± 0.080 % |
| q2_K         | WT 10k  |             2.96 | 8.652290 ±   0.055572  | 2.420657 ±   0.025587  | 0.331393 ±   0.001562 | -6.606 ± 0.046 %  | 18.790 ± 0.078 % |
| q2_K         | WT 100k |             2.96 | 8.641993 ±   0.055406  | 2.410359 ±   0.025495  | 0.331672 ±   0.001569 | -6.628 ± 0.047 %  | 18.856 ± 0.078 % |
| q2_K         | WT 10m  |             2.96 | 8.647825 ±   0.055610  | 2.416191 ±   0.025683  | 0.332223 ±   0.001572 | -6.500 ± 0.047 %  | 18.881 ± 0.078 % |
| q2_K         | WT 1m   |             2.96 | 8.674365 ±   0.055743  | 2.442732 ±   0.025843  | 0.335308 ±   0.001576 | -6.634 ± 0.047 %  | 19.009 ± 0.079 % |
| q2_K         | WT 1k   |             2.96 | 8.682605 ±   0.055916  | 2.450972 ±   0.026069  | 0.337093 ±   0.001596 | -6.596 ± 0.047 %  | 18.977 ± 0.079 % |
| q2_K_S       | WT 10m  |             2.96 | 9.323778 ±   0.061551  | 3.092145 ±   0.031914  | 0.403360 ±   0.001787 | -7.131 ± 0.049 %  | 20.050 ± 0.081 % |
| q2_K_S       | WT 1m   |             2.96 | 9.329321 ±   0.061378  | 3.097688 ±   0.031816  | 0.403590 ±   0.001797 | -7.289 ± 0.049 %  | 20.123 ± 0.081 % |
| q2_K_S       | WT 100k |             2.96 | 9.362973 ±   0.061740  | 3.131339 ±   0.032169  | 0.408367 ±   0.001802 | -7.198 ± 0.050 %  | 20.132 ± 0.081 % |
| q2_K_S       | WT 10k  |             2.96 | 9.376479 ±   0.062045  | 3.144846 ±   0.032464  | 0.408662 ±   0.001819 | -7.141 ± 0.050 %  | 20.120 ± 0.081 % |
| q2_K_S       | WT 1k   |             2.96 | 9.415200 ±   0.062475  | 3.183567 ±   0.032993  | 0.415865 ±   0.001846 | -7.153 ± 0.050 %  | 20.311 ± 0.082 % |
| iq2_S        | WT 10m  |             2.56 | 9.650781 ±   0.063209  | 3.419148 ±   0.034017  | 0.439197 ±   0.001976 | -8.319 ± 0.052 %  | 21.491 ± 0.083 % |
| q2_K         | None    |             2.96 | 9.751568 ±   0.063312  | 3.519934 ±   0.033863  | 0.445132 ±   0.001835 | -9.123 ± 0.051 %  | 21.421 ± 0.079 % |
| iq2_XS       | WT 10m  |             2.43 | 10.761424 ±   0.071056 | 4.529791 ±   0.042229  | 0.546290 ±   0.002133 | -10.576 ± 0.056 % | 23.872 ± 0.082 % |
| iq2_XXS      | WT 10m  |             2.24 | 14.091782 ±   0.098396 | 7.860148 ±   0.070752  | 0.812022 ±   0.002741 | -14.363 ± 0.065 % | 28.576 ± 0.084 % |
| iq1_M        | WT 10m  |             2.01 | 25.493722 ±   0.177903 | 19.262089 ±   0.152396 | 1.393084 ±   0.003529 | -24.672 ± 0.077 % | 38.287 ± 0.084 % |
| iq1_S        | WT 1m   |             1.88 | 58.097760 ±   0.438604 | 51.866126 ±   0.416604 | 2.211278 ±   0.004688 | -32.471 ± 0.087 % | 46.418 ± 0.085 % |
| iq1_S        | WT 1k   |             1.88 | 58.267851 ±   0.446208 | 52.036218 ±   0.424373 | 2.214858 ±   0.004778 | -31.880 ± 0.089 % | 46.330 ± 0.086 % |
| iq1_S        | WT 100k |             1.88 | 58.581498 ±   0.453145 | 52.349864 ±   0.431360 | 2.220834 ±   0.004818 | -32.261 ± 0.089 % | 46.002 ± 0.086 % |
| iq1_S        | WT 10m  |             1.88 | 60.694593 ±   0.471290 | 54.462959 ±   0.449644 | 2.254554 ±   0.004868 | -31.973 ± 0.088 % | 46.271 ± 0.086 % |
| iq1_S        | WT 10k  |             1.88 | 63.221324 ±   0.493077 | 56.989691 ±   0.471423 | 2.293527 ±   0.004885 | -32.261 ± 0.089 % | 46.562 ± 0.086 % |

There seems to be no consistent improvement from using more Wikitext tokens for the importance matrix.
K-quants score better on mean Δp than the legacy quants than e.g. KL divergence would suggest.

## LLaMA 2 vs. LLaMA 3 Quantization comparison

| Revision | f364eb6f           |
|:---------|:-------------------|
| Backend  | CUDA               |
| CPU      | AMD Epyc 7742      |
| GPU      | 1x NVIDIA RTX 4090 |

| Metric          |          L2 7b q2_K |          L3 8b q2_K |        L2 7b q4_K_M |        L3 8b q4_K_M |          L2 7b q6_K |          L3 8b q6_K |          L2 7b q8_0 |          L3 8b q8_0 |
|-----------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Mean PPL        | 5.794552 ± 0.032298 | 9.751568 ± 0.063312 | 5.877078 ± 0.032781 | 6.407115 ± 0.039119 | 5.808494 ± 0.032425 | 6.253382 ± 0.038078 | 5.798542 ± 0.032366 | 6.234284 ± 0.037878 |
| Mean PPL ratio  | 1.107955 ± 0.001427 | 1.564849 ± 0.004525 | 1.014242 ± 0.000432 | 1.028160 ± 0.000723 | 1.002406 ± 0.000191 | 1.003490 ± 0.000296 | 1.000689 ± 0.000107 | 1.000425 ± 0.000161 |
| Mean ΔPPL       | 0.625552 ± 0.008725 | 3.519934 ± 0.033863 | 0.082526 ± 0.002530 | 0.175482 ± 0.004620 | 0.013941 ± 0.001110 | 0.021748 ± 0.001852 | 0.003990 ± 0.000624 | 0.002650 ± 0.001006 |
| PPL correlation |              97.36% |              89.62% |              99.71% |              99.34% |              99.94% |              99.88% |              99.98% |              99.96% |
| Mean KLD        | 0.108903 ± 0.000645 | 0.445132 ± 0.001835 | 0.012686 ± 0.000079 | 0.031273 ± 0.000238 | 0.002098 ± 0.000014 | 0.005452 ± 0.000035 | 0.000369 ± 0.000007 | 0.001355 ± 0.000006 |
| Mean Δp         |    -2.710 ± 0.023 % |    -9.123 ± 0.051 % |    -0.416 ± 0.008 % |    -0.596 ± 0.014 % |    -0.035 ± 0.003 % |    -0.007 ± 0.006 % |    -0.005 ± 0.002 % |    -0.019 ± 0.003 % |
| Maximum Δp      |             85.136% |             94.268% |             45.209% |             95.054% |             23.593% |             53.601% |             43.925% |             28.734% |
| 99.9% Δp        |             37.184% |             50.003% |             17.461% |             27.084% |              7.798% |             13.613% |              3.387% |              6.402% |
| 99.0% Δp        |             18.131% |             25.875% |              7.798% |             12.084% |              3.838% |              6.407% |              1.867% |              3.544% |
| Median Δp       |             -0.391% |             -2.476% |             -0.026% |             -0.024% |             -0.001% |              0.000% |             -0.000% |             -0.000% |
| 1.0% Δp         |            -39.762% |            -87.173% |            -11.433% |            -19.567% |             -4.222% |             -6.767% |             -1.862% |             -3.698% |
| 0.1% Δp         |            -79.002% |            -98.897% |            -26.433% |            -56.054% |             -9.091% |            -16.584% |             -3.252% |             -6.579% |
| Minimum Δp      |            -99.915% |            -99.965% |            -83.383% |            -98.699% |            -43.142% |            -68.487% |             -9.343% |            -24.301% |
| RMS Δp          |     9.762 ± 0.053 % |    21.421 ± 0.079 % |     3.252 ± 0.024 % |     5.519 ± 0.050 % |     1.339 ± 0.010 % |     2.295 ± 0.019 % |     0.618 ± 0.011 % |     1.198 ± 0.007 % |
| Same top p      |    85.584 ± 0.086 % |    71.138 ± 0.119 % |    94.665 ± 0.055 % |    91.901 ± 0.072 % |    97.520 ± 0.038 % |    96.031 ± 0.051 % |    98.846 ± 0.026 % |    97.674 ± 0.040 % |

## LLaMA 3 BF16 vs. FP16 comparison

| Revision | 83330d8c      |
|:---------|:--------------|
| Backend  | CPU           |
| CPU      | AMD Epyc 7742 |
| GPU      | N/A           |

Results were calculated with LLaMA 3 8b BF16 as `--kl-divergence-base` and LLaMA 3 8b FP16 as the `--model` for comparison.

| Metric                         |                    Value |
|--------------------------------|--------------------------|
| Mean PPL(Q)                    |      6.227711 ± 0.037833 |
| Mean PPL(base)                 |      6.225194 ± 0.037771 |
| Cor(ln(PPL(Q)), ln(PPL(base))) |                  99.990% |
| Mean ln(PPL(Q)/PPL(base))      |      0.000404 ± 0.000086 |
| Mean PPL(Q)/PPL(base)          |      1.000404 ± 0.000086 |
| Mean PPL(Q)-PPL(base)          |      0.002517 ± 0.000536 |
| Mean    KLD                    |  0.00002515 ± 0.00000020 |
| Maximum KLD                    |                 0.012206 |
| 99.9%   KLD                    |                 0.000799 |
| 99.0%   KLD                    |                 0.000222 |
| 99.0%   KLD                    |                 0.000222 |
| Median  KLD                    |                 0.000013 |
| 10.0%   KLD                    |                -0.000002 |
| 5.0%   KLD                     |                -0.000008 |
| 1.0%   KLD                     |                -0.000023 |
| Minimum KLD                    |                -0.000059 |
| Mean    Δp                     | -0.0000745 ± 0.0003952 % |
| Maximum Δp                     |                   4.186% |
| 99.9%   Δp                     |                   1.049% |
| 99.0%   Δp                     |                   0.439% |
| 95.0%   Δp                     |                   0.207% |
| 90.0%   Δp                     |                   0.125% |
| 75.0%   Δp                     |                   0.029% |
| Median  Δp                     |                   0.000% |
| 25.0%   Δp                     |                  -0.030% |
| 10.0%   Δp                     |                  -0.126% |
| 5.0%   Δp                      |                  -0.207% |
| 1.0%   Δp                      |                  -0.434% |
| 0.1%   Δp                      |                  -1.016% |
| Minimum Δp                     |                  -4.672% |
| RMS Δp                         |          0.150 ± 0.001 % |
| Same top p                     |         99.739 ± 0.013 % |

## Old Numbers

<details>
<summary>Llama 2 70B Scoreboard</summary>

| Quantization | Model size (GiB) | Perplexity | Delta to fp16 |
|--------------|------------------|------------|---------------|
| Q4_0         | 36.20            | 3.5550     | 3.61%         |
| Q4_1         | 40.20            | 3.5125     | 2.37%         |
| Q5_0         | 44.20            | 3.4744     | 1.26%         |
| Q2_K         | 27.27            | 3.7339     | 8.82%         |
| Q3_K_S       | 27.86            | 3.7019     | 7.89%         |
| Q3_K_M       | 30.83            | 3.5932     | 4.72%         |
| Q3_K_L       | 33.67            | 3.5617     | 3.80%         |
| Q4_K_S       | 36.39            | 3.4852     | 1.57%         |
| Q4_K_M       | 38.54            | 3.4725     | 1.20%         |
| Q5_K_S       | 44.20            | 3.4483     | 0.50%         |
| Q5_K_M       | 45.41            | 3.4451     | 0.40%         |
| Q6_K         | 52.70            | 3.4367     | 0.16%         |
| fp16         | 128.5            | 3.4313     | -             |

</details>
