REGRESION LINEAL VINOS
================

Tenemos 1000 vinos a los cuales les medimos 11 variables de interés: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates y alcohol. A su vez, cada uno de estos vinos fue calificado con una nota del 0 al 10 por unos especialistas (que al calificarlos tuvieron en cuenta ´unicamente su "experiencia sensorial"). Estos datos fueron guardados en la variable quality. El objetivo de este TP es poder explicar mediante una regresión lineal la variable respuesta quality en función de las 11 variables explicativas ya mencionadas.

``` r
# cargo los datos
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(glmnet)
```

    ## Loading required package: Matrix

    ## Loading required package: foreach

    ## Loaded glmnet 2.0-16

``` r
library(glmnetUtils)
```

    ## 
    ## Attaching package: 'glmnetUtils'

    ## The following objects are masked from 'package:glmnet':
    ## 
    ##     cv.glmnet, glmnet

``` r
vinos <- read.csv("vinos_1.csv",header=TRUE,sep=",")
```

PREPROCESAR
===========

``` r
boxplot(vinos)
```

![](REGRESION_VINOS2_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
reemplazar_outliers=function(x){
  # reemplaza los outliers por el valor del quantile .02 y .98 segun corresponda.
  qnt <- quantile(x,probs=c(.25,.75))
  alfa <- quantile(x,probs=c(.02,0.98))
  H<-1.5*IQR(x)
  #x[x< (qnt[1]-H)]<-alfa[1]
  #x[x>(qnt[2]+H)]<-alfa[2]
  x[x<alfa[1]]<-alfa[1]
  x[x>alfa[2]]<-alfa[2]
  return(x)
}
preprocesar=function(data){
  
  for (i in 1:(length(data))){
    data[,i]=reemplazar_outliers(data[,i])
    #data[,i]=as.numeric(scale(data[,i],center= TRUE, scale= TRUE))
  }
  return(data)
}

# reemplazo los outliers en todas las columnas
data <- preprocesar(vinos)
boxplot(data)
```

![](REGRESION_VINOS2_files/figure-markdown_github/unnamed-chunk-3-1.png)

MODELOS
=======

``` r
set.seed(42)
```

### SEPARO EN TRAIN Y TEST

``` r
Y <- vinos[,"quality"]
train_indices<-createDataPartition(Y,p=0.8,list=FALSE) # mantiene la proporción de las clases de Y
vinos_train<- data[train_indices,]
vinos_test <- data[-train_indices,]
vinos_test_original<-vinos[-train_indices,]
```

MODELO 4
--------

En este modelo, considero que el logaritmo(quality) depende linealmente de las variables.

``` r
# defino train control para k fold cross validation

train_control<-trainControl(method = "cv",number = 10)

# entreno el modelo lineal

vinos.fit <-train(log(quality) ~.,
                   data = vinos_train, trControl=train_control,method ="lm")                  

# medidas promedio de los 10 folds. 

vinos.fit$results
```

<script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["intercept"],"name":[1],"type":["lgl"],"align":["right"]},{"label":["RMSE"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["Rsquared"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["MAE"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["RMSESD"],"name":[5],"type":["dbl"],"align":["right"]},{"label":["RsquaredSD"],"name":[6],"type":["dbl"],"align":["right"]},{"label":["MAESD"],"name":[7],"type":["dbl"],"align":["right"]}],"data":[{"1":"TRUE","2":"0.1101781","3":"0.3287208","4":"0.08656691","5":"0.008269252","6":"0.05882813","7":"0.005154263","_rn_":"1"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>

``` r
# resumen del modelo final
modeloFinal4 <- vinos.fit$finalModel
summary(modeloFinal4)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.42354 -0.06405 -0.01055  0.08195  0.29668 
    ## 
    ## Coefficients:
    ##                        Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)           0.6763220  5.3006338   0.128 0.898504    
    ## fixed.acidity         0.0055761  0.0063135   0.883 0.377403    
    ## volatile.acidity     -0.1642856  0.0297165  -5.528 4.39e-08 ***
    ## citric.acid          -0.0766805  0.0347220  -2.208 0.027502 *  
    ## residual.sugar        0.0022164  0.0048619   0.456 0.648610    
    ## chlorides            -0.2474292  0.1338227  -1.849 0.064841 .  
    ## free.sulfur.dioxide   0.0004292  0.0006176   0.695 0.487317    
    ## total.sulfur.dioxide -0.0006679  0.0001876  -3.559 0.000394 ***
    ## density               0.6474854  5.4036698   0.120 0.904654    
    ## pH                   -0.0433704  0.0455949  -0.951 0.341789    
    ## sulphates             0.1396478  0.0287246   4.862 1.41e-06 ***
    ## alcohol               0.0531101  0.0062838   8.452  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.1095 on 789 degrees of freedom
    ## Multiple R-squared:  0.3402, Adjusted R-squared:  0.331 
    ## F-statistic: 36.98 on 11 and 789 DF,  p-value: < 2.2e-16

``` r
# MSE de train_vinos
MSE <- mean((vinos_train[,"quality"]-exp(modeloFinal4$fitted.values))^2)
MSE
```

    ## [1] 0.3638155

``` r
plot(modeloFinal4$fitted.values,modeloFinal4$residuals)
```

![](REGRESION_VINOS2_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
# MSE de test_vinos
MSE <- mean((vinos_test[,"quality"]-exp(predict(modeloFinal4,vinos_test)))^2)
MSE
```

    ## [1] 0.3889965

``` r
MSE <- mean((vinos_test_original[,"quality"]-exp(predict(modeloFinal4,vinos_test)))^2)
MSE
```

    ## [1] 0.4649471

MODELO 5
--------

El último modelo considera "volatile.acidity", "citric.acid", "chlorides", "total.sulfur.dioxide", "sulphates","alcohol","pH" y el producto de estas variables.

``` r
vinos_train5 <- vinos_train[,match(c("volatile.acidity","citric.acid","chlorides","total.sulfur.dioxide","sulphates","alcohol","pH","quality"),colnames(vinos))]
vinos_test5 <- vinos_test[,match(c("volatile.acidity","citric.acid","chlorides","total.sulfur.dioxide","sulphates","alcohol","pH","quality"),colnames(vinos))]
l=1
for (i in 1:6){
  for (j in (i+1):7){
    vinos_train5[,8+l]<-vinos_train5[,i]*vinos_train5[,j]
    vinos_test5[,8+l]<-vinos_test5[,i]*vinos_test5[,j]
    l<-l+1
  }
}


foldid=sample(1:10,size=length(vinos_train5[,"quality"]),replace=TRUE)
cv.alpha0 <- cv.glmnet(quality~ ., data=vinos_train5,foldid=foldid,alpha=0) # Ridge
cv.alpha05 <- cv.glmnet(quality~ ., data=vinos_train5,foldid=foldid,alpha=0.5)
cv.alpha1 <- cv.glmnet(quality~ ., data=vinos_train5,foldid=foldid,alpha=1) # Lasso


plot(cv.alpha0)
```

![](REGRESION_VINOS2_files/figure-markdown_github/unnamed-chunk-7-1.png)

``` r
plot(cv.alpha05)
```

![](REGRESION_VINOS2_files/figure-markdown_github/unnamed-chunk-7-2.png)

``` r
plot(cv.alpha1)
```

![](REGRESION_VINOS2_files/figure-markdown_github/unnamed-chunk-7-3.png)

Elijo el modelo con alpha=1 y lambda = lambda.min

``` r
#MSE vinos_train
MSE<-cv.alpha1$cvm[match(cv.alpha1$lambda.min,cv.alpha1$lambda)]
MSE
```

    ## [1] 0.362377

``` r
coef(cv.alpha1, s="lambda.min")
```

    ## 29 x 1 sparse Matrix of class "dgCMatrix"
    ##                                  1
    ## (Intercept)           4.8559915576
    ## volatile.acidity     -1.2468457786
    ## citric.acid           .           
    ## chlorides             .           
    ## total.sulfur.dioxide  .           
    ## sulphates             .           
    ## alcohol               0.1926087793
    ## pH                   -0.4327384361
    ## V9                   -0.6681885006
    ## V10                   2.4291441345
    ## V11                   0.0092494392
    ## V12                  -0.0604620765
    ## V13                   .           
    ## V14                   .           
    ## V15                  -1.6791738715
    ## V16                  -0.0001215758
    ## V17                   .           
    ## V18                   0.0175299191
    ## V19                   .           
    ## V20                  -0.0078867037
    ## V21                  -2.0652308478
    ## V22                   .           
    ## V23                   .           
    ## V24                  -0.0117138912
    ## V25                   .           
    ## V26                   .           
    ## V27                   0.1598622345
    ## V28                   0.0188375430
    ## V29                   .

El MSE para vinos\_test es

``` r
prediccion <-predict(cv.alpha1, newdata=vinos_test5, s = "lambda.min")
MSE<-mean((vinos_test5[,"quality"]-prediccion)^2)
MSE
```

    ## [1] 0.391939

``` r
prediccion <-predict(cv.alpha1, newdata=vinos_test5, s = "lambda.min")
MSE<-mean((vinos_test_original[,"quality"]-prediccion)^2)
MSE
```

    ## [1] 0.4609823

ELECCION DE MODELO
==================

como el MSE de vinos\_test para el modelo5 es el mas chico, elijo este como el modelo final

``` r
vinos_prediccion <- read.csv("vinos_2.csv",header=TRUE,sep=",")
preprocesa=function(data){
  
  for (i in 1:(length(data))){
    data[,i]=reemplazar_outliers(data[,i])
  }
  return(data)
}
vinos_prediccion <- preprocesa(vinos_prediccion)
x=c("volatile.acidity","citric.acid","chlorides","total.sulfur.dioxide","sulphates","alcohol","pH")

vinos_p <- vinos_prediccion[,x]
vinos_p[,"quality"]<-1:599 # pongo cualquier cosa solo para que tenga "quality"
l<-1
for (i in 1:6){
  for (j in (i+1):7){
    vinos_p[,8+l]<-vinos_p[,i]*vinos_p[,j]
    vinos_p[,8+l]<-vinos_p[,i]*vinos_p[,j]
    l<-l+1
  }
}
predichos <-predict(cv.alpha1, newdata=vinos_p, s = "lambda.min")
round(predichos,3);
```

    ##            1
    ##   [1,] 6.254
    ##   [2,] 6.046
    ##   [3,] 6.641
    ##   [4,] 6.479
    ##   [5,] 5.593
    ##   [6,] 6.479
    ##   [7,] 6.641
    ##   [8,] 6.643
    ##   [9,] 6.414
    ##  [10,] 5.781
    ##  [11,] 6.454
    ##  [12,] 5.858
    ##  [13,] 5.117
    ##  [14,] 5.392
    ##  [15,] 5.893
    ##  [16,] 6.470
    ##  [17,] 6.668
    ##  [18,] 6.226
    ##  [19,] 6.226
    ##  [20,] 5.629
    ##  [21,] 6.134
    ##  [22,] 6.134
    ##  [23,] 5.682
    ##  [24,] 6.373
    ##  [25,] 5.659
    ##  [26,] 5.458
    ##  [27,] 6.363
    ##  [28,] 5.657
    ##  [29,] 5.497
    ##  [30,] 5.659
    ##  [31,] 5.886
    ##  [32,] 5.849
    ##  [33,] 5.143
    ##  [34,] 5.641
    ##  [35,] 5.165
    ##  [36,] 6.000
    ##  [37,] 6.441
    ##  [38,] 4.963
    ##  [39,] 6.579
    ##  [40,] 6.098
    ##  [41,] 5.306
    ##  [42,] 5.535
    ##  [43,] 6.098
    ##  [44,] 6.109
    ##  [45,] 6.228
    ##  [46,] 6.011
    ##  [47,] 5.599
    ##  [48,] 5.625
    ##  [49,] 6.107
    ##  [50,] 6.030
    ##  [51,] 5.625
    ##  [52,] 5.353
    ##  [53,] 6.170
    ##  [54,] 6.790
    ##  [55,] 5.118
    ##  [56,] 5.118
    ##  [57,] 6.173
    ##  [58,] 5.152
    ##  [59,] 6.029
    ##  [60,] 6.173
    ##  [61,] 6.078
    ##  [62,] 6.584
    ##  [63,] 6.134
    ##  [64,] 6.382
    ##  [65,] 5.874
    ##  [66,] 5.542
    ##  [67,] 6.462
    ##  [68,] 6.546
    ##  [69,] 6.546
    ##  [70,] 5.590
    ##  [71,] 6.416
    ##  [72,] 5.063
    ##  [73,] 5.283
    ##  [74,] 5.555
    ##  [75,] 5.063
    ##  [76,] 5.785
    ##  [77,] 6.470
    ##  [78,] 5.966
    ##  [79,] 5.966
    ##  [80,] 6.069
    ##  [81,] 6.329
    ##  [82,] 6.069
    ##  [83,] 5.287
    ##  [84,] 6.242
    ##  [85,] 5.287
    ##  [86,] 5.391
    ##  [87,] 6.334
    ##  [88,] 6.318
    ##  [89,] 5.734
    ##  [90,] 5.734
    ##  [91,] 6.178
    ##  [92,] 6.102
    ##  [93,] 5.957
    ##  [94,] 6.656
    ##  [95,] 5.482
    ##  [96,] 5.702
    ##  [97,] 5.482
    ##  [98,] 5.280
    ##  [99,] 6.694
    ## [100,] 5.280
    ## [101,] 6.701
    ## [102,] 6.179
    ## [103,] 5.814
    ## [104,] 6.179
    ## [105,] 6.367
    ## [106,] 6.215
    ## [107,] 6.494
    ## [108,] 6.646
    ## [109,] 5.050
    ## [110,] 5.671
    ## [111,] 5.710
    ## [112,] 6.093
    ## [113,] 6.129
    ## [114,] 5.812
    ## [115,] 6.287
    ## [116,] 5.788
    ## [117,] 5.788
    ## [118,] 5.788
    ## [119,] 6.438
    ## [120,] 6.214
    ## [121,] 6.615
    ## [122,] 6.154
    ## [123,] 6.059
    ## [124,] 6.111
    ## [125,] 5.919
    ## [126,] 6.273
    ## [127,] 6.560
    ## [128,] 5.734
    ## [129,] 5.414
    ## [130,] 5.669
    ## [131,] 5.783
    ## [132,] 5.218
    ## [133,] 6.873
    ## [134,] 6.081
    ## [135,] 6.601
    ## [136,] 6.207
    ## [137,] 6.204
    ## [138,] 6.204
    ## [139,] 5.177
    ## [140,] 5.222
    ## [141,] 5.262
    ## [142,] 5.979
    ## [143,] 6.194
    ## [144,] 6.039
    ## [145,] 5.540
    ## [146,] 5.908
    ## [147,] 5.786
    ## [148,] 6.049
    ## [149,] 6.165
    ## [150,] 6.241
    ## [151,] 6.906
    ## [152,] 6.125
    ## [153,] 5.349
    ## [154,] 5.987
    ## [155,] 5.823
    ## [156,] 5.349
    ## [157,] 6.122
    ## [158,] 6.383
    ## [159,] 6.014
    ## [160,] 5.895
    ## [161,] 6.211
    ## [162,] 5.866
    ## [163,] 6.517
    ## [164,] 5.360
    ## [165,] 5.360
    ## [166,] 5.555
    ## [167,] 5.844
    ## [168,] 6.675
    ## [169,] 6.219
    ## [170,] 6.038
    ## [171,] 6.048
    ## [172,] 5.779
    ## [173,] 6.557
    ## [174,] 5.313
    ## [175,] 5.313
    ## [176,] 5.865
    ## [177,] 5.419
    ## [178,] 6.375
    ## [179,] 5.683
    ## [180,] 6.138
    ## [181,] 6.138
    ## [182,] 5.939
    ## [183,] 5.885
    ## [184,] 5.213
    ## [185,] 5.514
    ## [186,] 5.922
    ## [187,] 5.739
    ## [188,] 5.922
    ## [189,] 5.514
    ## [190,] 4.941
    ## [191,] 6.312
    ## [192,] 5.372
    ## [193,] 6.638
    ## [194,] 5.372
    ## [195,] 5.214
    ## [196,] 5.284
    ## [197,] 5.212
    ## [198,] 5.461
    ## [199,] 6.048
    ## [200,] 5.212
    ## [201,] 5.461
    ## [202,] 6.057
    ## [203,] 6.294
    ## [204,] 5.157
    ## [205,] 6.121
    ## [206,] 6.121
    ## [207,] 6.121
    ## [208,] 5.350
    ## [209,] 6.121
    ## [210,] 6.316
    ## [211,] 5.632
    ## [212,] 5.349
    ## [213,] 5.632
    ## [214,] 6.127
    ## [215,] 5.798
    ## [216,] 6.068
    ## [217,] 5.233
    ## [218,] 6.336
    ## [219,] 5.965
    ## [220,] 6.144
    ## [221,] 6.157
    ## [222,] 6.157
    ## [223,] 5.209
    ## [224,] 6.541
    ## [225,] 6.105
    ## [226,] 5.260
    ## [227,] 5.288
    ## [228,] 5.502
    ## [229,] 6.324
    ## [230,] 5.309
    ## [231,] 6.374
    ## [232,] 5.556
    ## [233,] 5.309
    ## [234,] 5.509
    ## [235,] 5.884
    ## [236,] 5.782
    ## [237,] 5.500
    ## [238,] 5.884
    ## [239,] 5.131
    ## [240,] 5.881
    ## [241,] 5.163
    ## [242,] 5.484
    ## [243,] 6.275
    ## [244,] 5.183
    ## [245,] 5.390
    ## [246,] 5.659
    ## [247,] 5.346
    ## [248,] 5.659
    ## [249,] 6.047
    ## [250,] 5.746
    ## [251,] 5.746
    ## [252,] 5.377
    ## [253,] 5.326
    ## [254,] 5.294
    ## [255,] 5.747
    ## [256,] 5.519
    ## [257,] 5.141
    ## [258,] 5.723
    ## [259,] 5.851
    ## [260,] 5.851
    ## [261,] 5.147
    ## [262,] 5.438
    ## [263,] 5.419
    ## [264,] 5.101
    ## [265,] 6.221
    ## [266,] 5.581
    ## [267,] 5.581
    ## [268,] 6.506
    ## [269,] 5.276
    ## [270,] 6.437
    ## [271,] 6.510
    ## [272,] 5.953
    ## [273,] 5.783
    ## [274,] 5.189
    ## [275,] 5.776
    ## [276,] 5.146
    ## [277,] 6.246
    ## [278,] 5.392
    ## [279,] 5.146
    ## [280,] 6.302
    ## [281,] 5.638
    ## [282,] 5.638
    ## [283,] 5.761
    ## [284,] 5.414
    ## [285,] 5.795
    ## [286,] 5.710
    ## [287,] 6.409
    ## [288,] 6.107
    ## [289,] 5.420
    ## [290,] 5.420
    ## [291,] 5.694
    ## [292,] 5.818
    ## [293,] 6.371
    ## [294,] 5.182
    ## [295,] 5.818
    ## [296,] 5.251
    ## [297,] 5.251
    ## [298,] 6.162
    ## [299,] 6.065
    ## [300,] 5.262
    ## [301,] 6.103
    ## [302,] 5.807
    ## [303,] 6.392
    ## [304,] 5.915
    ## [305,] 5.023
    ## [306,] 5.230
    ## [307,] 5.281
    ## [308,] 5.601
    ## [309,] 5.281
    ## [310,] 5.203
    ## [311,] 5.230
    ## [312,] 6.365
    ## [313,] 5.167
    ## [314,] 5.471
    ## [315,] 5.450
    ## [316,] 5.242
    ## [317,] 5.973
    ## [318,] 6.221
    ## [319,] 5.242
    ## [320,] 4.933
    ## [321,] 5.315
    ## [322,] 5.973
    ## [323,] 6.137
    ## [324,] 6.142
    ## [325,] 5.710
    ## [326,] 5.710
    ## [327,] 5.710
    ## [328,] 5.710
    ## [329,] 5.252
    ## [330,] 5.238
    ## [331,] 5.238
    ## [332,] 4.901
    ## [333,] 5.515
    ## [334,] 5.138
    ## [335,] 5.274
    ## [336,] 6.326
    ## [337,] 5.294
    ## [338,] 5.294
    ## [339,] 5.294
    ## [340,] 5.646
    ## [341,] 5.646
    ## [342,] 5.646
    ## [343,] 5.555
    ## [344,] 5.646
    ## [345,] 5.894
    ## [346,] 5.619
    ## [347,] 5.823
    ## [348,] 5.134
    ## [349,] 5.134
    ## [350,] 5.628
    ## [351,] 5.311
    ## [352,] 5.885
    ## [353,] 5.381
    ## [354,] 5.381
    ## [355,] 5.433
    ## [356,] 5.529
    ## [357,] 5.522
    ## [358,] 5.665
    ## [359,] 5.109
    ## [360,] 5.683
    ## [361,] 5.684
    ## [362,] 5.220
    ## [363,] 5.683
    ## [364,] 5.073
    ## [365,] 5.878
    ## [366,] 5.457
    ## [367,] 5.254
    ## [368,] 5.563
    ## [369,] 5.118
    ## [370,] 5.425
    ## [371,] 5.030
    ## [372,] 6.419
    ## [373,] 5.030
    ## [374,] 5.011
    ## [375,] 5.285
    ## [376,] 5.267
    ## [377,] 5.127
    ## [378,] 5.927
    ## [379,] 5.499
    ## [380,] 5.770
    ## [381,] 5.770
    ## [382,] 4.984
    ## [383,] 5.242
    ## [384,] 5.242
    ## [385,] 5.161
    ## [386,] 4.985
    ## [387,] 5.350
    ## [388,] 5.350
    ## [389,] 5.487
    ## [390,] 5.355
    ## [391,] 6.207
    ## [392,] 5.646
    ## [393,] 5.370
    ## [394,] 5.402
    ## [395,] 5.113
    ## [396,] 5.376
    ## [397,] 5.388
    ## [398,] 5.246
    ## [399,] 5.483
    ## [400,] 5.916
    ## [401,] 5.245
    ## [402,] 5.245
    ## [403,] 6.625
    ## [404,] 6.382
    ## [405,] 5.622
    ## [406,] 6.541
    ## [407,] 6.630
    ## [408,] 5.969
    ## [409,] 6.920
    ## [410,] 5.969
    ## [411,] 5.489
    ## [412,] 5.820
    ## [413,] 6.630
    ## [414,] 5.560
    ## [415,] 5.880
    ## [416,] 5.581
    ## [417,] 5.880
    ## [418,] 6.508
    ## [419,] 5.451
    ## [420,] 5.242
    ## [421,] 5.451
    ## [422,] 5.339
    ## [423,] 6.134
    ## [424,] 5.833
    ## [425,] 5.817
    ## [426,] 5.817
    ## [427,] 6.389
    ## [428,] 5.965
    ## [429,] 5.774
    ## [430,] 6.341
    ## [431,] 5.688
    ## [432,] 5.397
    ## [433,] 6.579
    ## [434,] 5.838
    ## [435,] 4.901
    ## [436,] 4.901
    ## [437,] 5.033
    ## [438,] 5.278
    ## [439,] 5.436
    ## [440,] 5.727
    ## [441,] 6.275
    ## [442,] 5.174
    ## [443,] 5.565
    ## [444,] 5.802
    ## [445,] 5.737
    ## [446,] 5.183
    ## [447,] 5.565
    ## [448,] 5.469
    ## [449,] 5.465
    ## [450,] 6.310
    ## [451,] 6.275
    ## [452,] 6.248
    ## [453,] 6.119
    ## [454,] 5.044
    ## [455,] 5.955
    ## [456,] 5.479
    ## [457,] 5.531
    ## [458,] 5.044
    ## [459,] 5.902
    ## [460,] 6.812
    ## [461,] 5.690
    ## [462,] 5.271
    ## [463,] 5.556
    ## [464,] 5.625
    ## [465,] 5.388
    ## [466,] 5.388
    ## [467,] 5.525
    ## [468,] 5.391
    ## [469,] 5.525
    ## [470,] 5.126
    ## [471,] 5.310
    ## [472,] 5.996
    ## [473,] 6.083
    ## [474,] 5.666
    ## [475,] 4.924
    ## [476,] 6.455
    ## [477,] 4.924
    ## [478,] 6.455
    ## [479,] 5.190
    ## [480,] 5.947
    ## [481,] 5.460
    ## [482,] 5.947
    ## [483,] 5.421
    ## [484,] 6.004
    ## [485,] 5.443
    ## [486,] 5.262
    ## [487,] 5.447
    ## [488,] 5.835
    ## [489,] 5.856
    ## [490,] 5.992
    ## [491,] 6.524
    ## [492,] 5.856
    ## [493,] 5.841
    ## [494,] 5.175
    ## [495,] 5.844
    ## [496,] 6.087
    ## [497,] 5.175
    ## [498,] 5.826
    ## [499,] 5.359
    ## [500,] 5.826
    ## [501,] 5.244
    ## [502,] 5.045
    ## [503,] 5.310
    ## [504,] 5.965
    ## [505,] 5.948
    ## [506,] 5.326
    ## [507,] 5.566
    ## [508,] 5.948
    ## [509,] 6.055
    ## [510,] 6.364
    ## [511,] 5.644
    ## [512,] 5.409
    ## [513,] 5.443
    ## [514,] 5.564
    ## [515,] 4.982
    ## [516,] 4.987
    ## [517,] 6.061
    ## [518,] 5.831
    ## [519,] 5.689
    ## [520,] 5.402
    ## [521,] 5.831
    ## [522,] 5.118
    ## [523,] 6.061
    ## [524,] 5.565
    ## [525,] 5.704
    ## [526,] 5.547
    ## [527,] 5.534
    ## [528,] 5.904
    ## [529,] 5.614
    ## [530,] 5.409
    ## [531,] 6.154
    ## [532,] 5.410
    ## [533,] 5.688
    ## [534,] 5.127
    ## [535,] 6.001
    ## [536,] 5.399
    ## [537,] 5.566
    ## [538,] 5.546
    ## [539,] 5.985
    ## [540,] 5.575
    ## [541,] 5.968
    ## [542,] 6.117
    ## [543,] 5.482
    ## [544,] 5.712
    ## [545,] 6.419
    ## [546,] 5.450
    ## [547,] 5.607
    ## [548,] 5.876
    ## [549,] 5.658
    ## [550,] 6.280
    ## [551,] 5.195
    ## [552,] 5.188
    ## [553,] 5.778
    ## [554,] 5.208
    ## [555,] 5.492
    ## [556,] 5.736
    ## [557,] 5.097
    ## [558,] 5.492
    ## [559,] 4.974
    ## [560,] 5.254
    ## [561,] 5.254
    ## [562,] 5.254
    ## [563,] 5.373
    ## [564,] 5.373
    ## [565,] 5.373
    ## [566,] 5.926
    ## [567,] 6.160
    ## [568,] 5.373
    ## [569,] 5.237
    ## [570,] 5.885
    ## [571,] 6.486
    ## [572,] 6.063
    ## [573,] 5.063
    ## [574,] 6.140
    ## [575,] 5.428
    ## [576,] 6.132
    ## [577,] 6.146
    ## [578,] 5.845
    ## [579,] 5.853
    ## [580,] 5.816
    ## [581,] 6.238
    ## [582,] 5.816
    ## [583,] 5.685
    ## [584,] 5.279
    ## [585,] 6.446
    ## [586,] 6.280
    ## [587,] 6.234
    ## [588,] 5.727
    ## [589,] 6.219
    ## [590,] 5.045
    ## [591,] 6.261
    ## [592,] 5.658
    ## [593,] 5.956
    ## [594,] 5.491
    ## [595,] 5.535
    ## [596,] 5.928
    ## [597,] 5.956
    ## [598,] 5.481
    ## [599,] 5.982

``` r
write.csv(round(predichos,3), file = "predichos.csv",row.names = FALSE)
```
