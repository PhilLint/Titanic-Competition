DM Group 34
================
Pl Le Fc
4/18/2019

for this task, we chose mlr package in R, as it offers a modular structure, which allows to apply various kinds of classifiers with numereous advanced techniques to be easily applied. We also use the tidyverse data format, as it allows for elegant handling of matrix computations not requiring loops but functions with the %&lt;% pipeline.

For more information on the MLR package there exists a turorial: <https://mlr-org.github.io/mlr-tutorial/release/html/index.html>

``` r
library(tidyverse) # data format / manipulation
library(mlr)       # Best and most complete ML package 
library(knitr)     # 
library(xgboost)   # xgboost classifier
library(randomForest) # random forest classifier
library(gbm)
library(ggplot2)
library(data.table)
```

Load the datasets given on kaggle.

``` r
train_raw <- read_csv("./train.csv")
test_raw <- read_csv("./test.csv")
```

Data Preprocessing - Cleaning / Overview
----------------------------------------

Merging the train and test set eases the preprocessing steps as they only have to be applied on one object, which then can be reformatted thanks to an added column stating whether its data from the train or test set.

``` r
train <- train_raw %>% mutate(data = "train")
test <- test_raw %>% mutate(data = "test")
# same columns -> easily binded by row
full_data <- bind_rows(train, test)
```

As mentioned above, the mlr package includes many handy features. First and foremost handling the ML tasks in a very handy framework. Furthermore, functions, such as `summarizeColumns()` allow for a first overview on the data:

``` r
full_data <- full_data %>% mutate_each(funs(factor), Survived, Pclass, Name, Sex, Ticket, Cabin, Embarked, SibSp, Parch)
summarizeColumns(full_data) %>% kable(digits = 2)
```

| name        | type      |    na|    mean|    disp|  median|     mad|     min|      max|  nlevs|
|:------------|:----------|-----:|-------:|-------:|-------:|-------:|-------:|--------:|------:|
| PassengerId | integer   |     0|  655.00|  378.02|  655.00|  484.81|    1.00|  1309.00|      0|
| Survived    | factor    |   418|      NA|      NA|      NA|      NA|  342.00|   549.00|      2|
| Pclass      | factor    |     0|      NA|    0.46|      NA|      NA|  277.00|   709.00|      3|
| Name        | factor    |     0|      NA|    1.00|      NA|      NA|    1.00|     2.00|   1307|
| Sex         | factor    |     0|      NA|    0.36|      NA|      NA|  466.00|   843.00|      2|
| Age         | numeric   |   263|   29.88|   14.41|   28.00|   11.86|    0.17|    80.00|      0|
| SibSp       | factor    |     0|      NA|    0.32|      NA|      NA|    6.00|   891.00|      7|
| Parch       | factor    |     0|      NA|    0.23|      NA|      NA|    2.00|  1002.00|      8|
| Ticket      | factor    |     0|      NA|    0.99|      NA|      NA|    1.00|    11.00|    929|
| Fare        | numeric   |     1|   33.30|   51.76|   14.45|   10.24|    0.00|   512.33|      0|
| Cabin       | factor    |  1014|      NA|      NA|      NA|      NA|    1.00|     6.00|    186|
| Embarked    | factor    |     2|      NA|      NA|      NA|      NA|  123.00|   914.00|      3|
| data        | character |     0|      NA|    0.32|      NA|      NA|  418.00|   891.00|      2|

A lot of missing values in the entire dataset.

``` r
frequencies = full_data %>%  select(Survived,Pclass, Sex, Embarked, SibSp, Parch) %>% 
  gather(.,"var","value") %>% count(var, value) 
frequencies
```

    ## # A tibble: 27 x 3
    ##    var      value     n
    ##    <chr>    <chr> <int>
    ##  1 Embarked C       270
    ##  2 Embarked Q       123
    ##  3 Embarked S       914
    ##  4 Embarked <NA>      2
    ##  5 Parch    0      1002
    ##  6 Parch    1       170
    ##  7 Parch    2       113
    ##  8 Parch    3         8
    ##  9 Parch    4         6
    ## 10 Parch    5         6
    ## # ... with 17 more rows

61 % of the passengers did not survive, 342 did and also 418 unknowns. About two thirds of all passengers were male, most people had a third class ticket and also most people embarked in southhampton.

Correlations / important features
---------------------------------

Difference in survival rate between men and women?

``` r
full_data %>% group_by(Sex) %>% summarise(prob = mean(as.integer(Survived)-1, na.rm=T)) %>% print()
```

    ## # A tibble: 2 x 2
    ##   Sex     prob
    ##   <fct>  <dbl>
    ## 1 female 0.742
    ## 2 male   0.189

Yes, only 18% of men survived, whereas 74% of women survived.

Difference in survival rate between Classes?

``` r
full_data %>% group_by(Pclass) %>% summarise(prob = mean(as.integer(Survived)-1, na.rm=T)) %>% print()
```

    ## # A tibble: 3 x 2
    ##   Pclass  prob
    ##   <fct>  <dbl>
    ## 1 1      0.630
    ## 2 2      0.473
    ## 3 3      0.242

Yes, the lower the class, the higher the survival rate.

Difference in survival rate between Classes?

``` r
full_data %>% group_by(Embarked) %>% summarise(prob = mean(as.integer(Survived)-1, na.rm=T)) %>% print()
```

    ## # A tibble: 4 x 2
    ##   Embarked  prob
    ##   <fct>    <dbl>
    ## 1 C        0.554
    ## 2 Q        0.390
    ## 3 S        0.337
    ## 4 <NA>     1

Difference in survival rate between ages?

``` r
p1 <- full_data %>% filter(!is.na(Survived)) %>% ggplot(aes(Age,fill=Survived)) + geom_density(alpha=.5)

p2 <- full_data %>% filter(!is.na(Survived)) %>% filter(Fare < 300) %>% ggplot(aes(Fare,fill=Survived)) + geom_density(alpha=.5)

grid.arrange(p1,p2,nrow=1)
```

![](Titanic_competition_files/figure-markdown_github/unnamed-chunk-7-1.png)

Feature Engineering
-------------------

Too little data is a common problem for Data Analysis tasks, especially if there are many missing values as well. A way to deal with this in order to leverage the given observations for the most, a technique called Feature Engineering comes into play. In other words, we try to get as much information as possible out of the given data, even if it means to change the variables.

We start with the extraction of the Title of a person, which is derivable from the name.

``` r
head(full_data$Name)
```

    ## [1] Braund, Mr. Owen Harris                            
    ## [2] Cumings, Mrs. John Bradley (Florence Briggs Thayer)
    ## [3] Heikkinen, Miss. Laina                             
    ## [4] Futrelle, Mrs. Jacques Heath (Lily May Peel)       
    ## [5] Allen, Mr. William Henry                           
    ## [6] Moran, Mr. James                                   
    ## 1307 Levels: Abbing, Mr. Anthony ... Zimmerman, Mr. Leo

``` r
# Names follow structure e.g. : xxx, MR., xxx 
# -> delete everything before , and after . to get title
full_data$Title <- gsub('(.*, )|([.].*)', '', full_data$Name) 

full_data %>%
    group_by(Title, Sex) %>%
    summarize(n = n()) %>%
    arrange(Title)
```

    ## # A tibble: 19 x 3
    ## # Groups:   Title [18]
    ##    Title        Sex        n
    ##    <chr>        <fct>  <int>
    ##  1 Capt         male       1
    ##  2 Col          male       4
    ##  3 Don          male       1
    ##  4 Dona         female     1
    ##  5 Dr           female     1
    ##  6 Dr           male       7
    ##  7 Jonkheer     male       1
    ##  8 Lady         female     1
    ##  9 Major        male       2
    ## 10 Master       male      61
    ## 11 Miss         female   260
    ## 12 Mlle         female     2
    ## 13 Mme          female     1
    ## 14 Mr           male     757
    ## 15 Mrs          female   197
    ## 16 Ms           female     2
    ## 17 Rev          male       8
    ## 18 Sir          male       1
    ## 19 the Countess female     1

Only few categories with more then 10 instances, which is why we throw them together. Also, a lot of categories describe the same, such as Miss and Ms. Therefore, merging several categories as well as gathering all rare ones in a separate category is done.

``` r
full_data$Title[full_data$Title %in% c('Mlle', 'Ms')]  <- 'Miss'
full_data$Title[full_data$Title == 'Mme']   <- 'Mrs'

rare_titles  <- c('Capt', 'Col', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'the Countess', 'Don')
full_data$Title[full_data$Title %in% rare_titles]  <- 'rare'

full_data <- full_data %>% mutate_each(funs(factor), Title)


full_data %>% group_by(Title) %>% summarise(prob = mean(as.integer(Survived), na.rm=T)) %>% print()
```

    ## # A tibble: 5 x 2
    ##   Title   prob
    ##   <fct>  <dbl>
    ## 1 Master  1.58
    ## 2 Miss    1.70
    ## 3 Mr      1.16
    ## 4 Mrs     1.79
    ## 5 rare    1.35

There are noticeable differences among the titles. Most noticeably, Miss and Mrs survived to a much higher rate, than Mr.

We could also investigate the size of families on the survival rate. However, there are a lot of unique surnames, which does not indicate a big influence of families or their sizes:

``` r
full_data$Surname <- full_data$Name %>%
    gsub(',.*$','', .) 

cat('Number of unique surnames: ',nlevels(factor(full_data$Surname)))
```

    ## Number of unique surnames:  875

### Feature selection

We do not apply proper feature selection as in applying significance tests for a model with the respective vs without. However, we delete variables, that very likely do not contribute greatly. For instance Cabine has186 categories with very few instances. ALso the PassengerId, the Name, the Ticket and Surname are disregarded for similar reasons.

``` r
length(table(full_data$Cabin))
```

    ## [1] 186

``` r
full_data <- full_data %>%
    select(-c(PassengerId, Name, Ticket, Cabin, Surname)) 
head(full_data)
```

    ## # A tibble: 6 x 10
    ##   Survived Pclass Sex      Age SibSp Parch  Fare Embarked data  Title
    ##   <fct>    <fct>  <fct>  <dbl> <fct> <fct> <dbl> <fct>    <chr> <fct>
    ## 1 0        3      male      22 1     0      7.25 S        train Mr   
    ## 2 1        1      female    38 1     0     71.3  C        train Mrs  
    ## 3 1        3      female    26 0     0      7.92 S        train Miss 
    ## 4 1        1      female    35 1     0     53.1  S        train Mrs  
    ## 5 0        3      male      35 0     0      8.05 S        train Mr   
    ## 6 0        3      male      NA 0     0      8.46 Q        train Mr

### Missing Value Imputation

NA values are indesirable, as we either disregard the respective observations and thus decrease the amount of data drstically or have to complete the data artificially. It depends on the variables and amount of missing data. The necessity of Imputation can be assesed by comparing a models performance with and without Imputation. Looking at the amount of missing age and survival values, we decide to impute missing values. There exist quite a few techniques, such as inserting mean/median or predicted (e.g. linear regression on other variables) for numerical values or mode/most or also predicting the values with another classifier for the appearing instances for categorial variables.

Mlr offers an easy way to compute imputation:

``` r
# Impute missing values by simple mean/mode imputation
full_data_imp1 <- impute(
  full_data,
  classes = list(
    factor = imputeMode(),
    integer = imputeMean(),
    numeric = imputeMean()
  )
)

full_imp1 <- full_data_imp1$data
```

### Factor to dummy transformation

Mlr requires one-hot encoded categorial variables. Therefore, for instance Sex is split up into two binary variables "male" and "female", which are either 0 or 1. This procedure increase the speed of the algorithms.

``` r
# with imputation
full_data_imp_dummy <- createDummyFeatures(
  full_imp1, 
  cols = c(
    "Pclass",
    "Sex",
    "Embarked", 
    "Title",
    "Parch",
    "SibSp"
  )
)

#summary(full_data_imp_dummy) 
```

Preprocessing is finished, which means we can split the dataset back to train and test based on the column data for the final Kaggle evaluation. For our purposes, we split the trainset 80/20 and evaluate on the 20% of test observations.

``` r
# with imputation
train <- full_data_imp_dummy %>%
  filter(data == "train") %>%
  select(-data)

final_test <- full_data_imp_dummy %>%
  filter(data == "test") %>%
  select(-data)

n = nrow(train)
# usually 80% train is good 
n_train = as.integer(0.8*n)
set.seed(42)
train_ids = sample(1:n, n_train)
#length(train_ids)
train_train = train[train_ids,]
train_test  = train[-train_ids,]
```

Fitting Logreg/ Decision Tree/ Random Forest/ XGBoost
-----------------------------------------------------

Mlr package requires a Task to be specified, which involves the train / test split.

``` r
trainTask <- makeClassifTask(data = train_train, target = "Survived", positive = 1)
testTask <- makeClassifTask(data = train_test, target = "Survived")
# for Kaggle submission take entire trainset
final_test <- makeClassifTask(data = final_test, target = "Survived")
final_train <- makeClassifTask(data = train, target = "Survived")
```

### Logistic regression

We start with a first baseline model, logistic regression.

``` r
logreg_learner = makeLearner("classif.logreg")
# Train model
logreg_model <- train(logreg_learner, task = trainTask)
# Test model
predictions <- predict(logreg_model, testTask)
# our setup
performance(predictions, measures = list(ppv, acc))
```

    ##       ppv       acc 
    ## 0.6285714 0.7709497

``` r
# KAGGLE
# retrain with full dataset
logreg_model <- train(logreg_learner, final_train)
final_predictions <- predict(logreg_model, final_test)
# prediction file
logreg_model_prediction <- final_predictions$data %>%
  select(PassengerID = id, Survived = response) %>%
  mutate(PassengerID = test_raw$PassengerId)

logreg_model_prediction$Survived = as.integer(logreg_model_prediction$Survived)-1
write.csv(logreg_model_prediction,"logreg_prediction.csv", row.names=F)
```

Precision of 0.564 and accuracy of 0.687 (bad). To demonstrate the bad performance of logreg, we upload it to kaggle too. Basic logistic regression yielded a Score of 0.70813, which is pretty low (rank 10183). Considering, this is the most basic classification model, it makes sense.

### Decision Tree

The next baseline model, is an ordinary decision tree, also called cart.

``` r
rpart_learner = makeLearner("classif.rpart")
# Train model
rpart_model <- train(rpart_learner, task = trainTask)
# Test model
predictions <- predict(rpart_model, testTask)
final_predictions <- predict(rpart_model, final_test)
# on our setup
performance(predictions, measures = list(ppv, acc))
```

    ##       ppv       acc 
    ## 0.7368421 0.8212291

``` r
# KAGGLE
# retrain with full dataset
rpart_model <- train(rpart_learner, final_train)
final_predictions <- predict(rpart_model, final_test)
# prediction file
rpart_model_prediction <- final_predictions$data %>%
  select(PassengerID = id, Survived = response) %>%
  mutate(PassengerID = test_raw$PassengerId)

rpart_model_prediction$Survived = as.integer(rpart_model_prediction$Survived)-1
write.csv(rpart_model_prediction,"rpart_prediction.csv", row.names=F)
```

This method yielded a much higher precision, as well as accuracy value. Even without parameter tuning or tuning efforts, 75% of test data are assessed correctly.

KAGGLE: Decision Tree yielded a Score of 0.7799, which is already much better (rank 5053). Considering, decision trees are performing better, if there are a lot of categorial variables, this result also makes sense.

### Random Forest

The next baseline model, is an extension to ordinary decision trees, called random Forest, which incorporates randomness in the sense that for splits not all variables can be used, but with a random chance a few are disregarded per split. Also, bagging is used in the sense that many of these trees are constructed to aggregate each trees decision to a global decision based on the majority vote of these trees.

``` r
rforest_learner = makeLearner("classif.randomForest")
# Train model
rforest_model <- train(rforest_learner, task = trainTask)
# Test model
predictions <- predict(rforest_model, testTask)

# average performance over 5 runs
perf = list()
for(i in 1:5){
  set.seed(i)
  rforest_model <- train(rforest_learner, task = trainTask)
  predictions <- predict(rforest_model, testTask)
  perf[i] <- list(performance(predictions, measures = list(ppv, acc)))
}

mean(as.numeric(unname(as.data.table(perf)[1,])))
```

    ## [1] 0.7464629

``` r
mean(as.numeric(unname(as.data.table(perf)[2,])))
```

    ## [1] 0.8223464

``` r
rforest_results = c(mean(as.numeric(unname(as.data.table(perf)[1,]))), mean(as.numeric(unname(as.data.table(perf)[2,]))))

# KAGGLE
# retrain with full dataset
rforest_model <- train(rforest_learner, final_train)
final_predictions <- predict(rforest_model, final_test)
# prediction file
rforest_prediction <- final_predictions$data %>%
  select(PassengerID = id, Survived = response) %>%
  mutate(PassengerID = test_raw$PassengerId)

rforest_prediction$Survived = as.integer(rforest_prediction$Survived)-1
write.csv(rforest_prediction,"rforest_prediction.csv", row.names=F)
```

### XGBOOST

Another advanced model is the so called XGboost classifier.

``` r
xgb_learner = makeLearner("classif.xgboost")
# Train model
xgb_model <- train(xgb_learner, task = trainTask)

predictions <- predict(xgb_model, testTask)
final_predictions <- predict(xgb_model, final_test)

# prediction file
xgboost_prediction <- final_predictions$data %>%
  select(PassengerID = id, Survived = response) %>%
  mutate(PassengerID = test_raw$PassengerId)

xgboost_prediction$Survived = as.integer(xgboost_prediction$Survived)-1
write.csv(xgboost_prediction,"xgboost_prediction.csv", row.names=F)

perf = list()
for(i in 1:5){
  set.seed(i)
  xgb_model <- train(xgb_learner, task = trainTask)
  predictions <- predict(xgb_model, testTask)
  perf[i] <- list(performance(predictions, measures = list(ppv, acc)))
}

mean(as.numeric(unname(as.data.table(perf)[1,])))
```

    ## [1] 0.7333333

``` r
mean(as.numeric(unname(as.data.table(perf)[2,])))
```

    ## [1] 0.8268156

``` r
xgb_results = c(mean(as.numeric(unname(as.data.table(perf)[1,]))), mean(as.numeric(unname(as.data.table(perf)[2,]))))


# KAGGLE
# Re-train parameters using tuned hyperparameters (and full training set)
xgb_model <- train(xgb_learner, final_train)
final_predictions <- predict(xgb_model, final_test)

# prediction file
xgboost_prediction <- final_predictions$data %>%
  select(PassengerID = id, Survived = response) %>%
  mutate(PassengerID = test_raw$PassengerId)

xgboost_prediction$Survived = as.integer(xgboost_prediction$Survived)-1
write.csv(xgboost_prediction,"xgboost_prediction.csv", row.names=F)
```

Hyper-parameter Tuning
----------------------

First, the Random Forest classifier is tuned over the number of trees created, number of variables sampled as candidates each split

``` r
# To see all the parameters of the xgboost classifier
getParamSet("classif.randomForest")
```

    ##                      Type  len   Def   Constr Req Tunable Trafo
    ## ntree             integer    -   500 1 to Inf   -    TRUE     -
    ## mtry              integer    -     - 1 to Inf   -    TRUE     -
    ## replace           logical    -  TRUE        -   -    TRUE     -
    ## classwt     numericvector <NA>     - 0 to Inf   -    TRUE     -
    ## cutoff      numericvector <NA>     -   0 to 1   -    TRUE     -
    ## strata            untyped    -     -        -   -   FALSE     -
    ## sampsize    integervector <NA>     - 1 to Inf   -    TRUE     -
    ## nodesize          integer    -     1 1 to Inf   -    TRUE     -
    ## maxnodes          integer    -     - 1 to Inf   -    TRUE     -
    ## importance        logical    - FALSE        -   -    TRUE     -
    ## localImp          logical    - FALSE        -   -    TRUE     -
    ## proximity         logical    - FALSE        -   -   FALSE     -
    ## oob.prox          logical    -     -        -   Y   FALSE     -
    ## norm.votes        logical    -  TRUE        -   -   FALSE     -
    ## do.trace          logical    - FALSE        -   -   FALSE     -
    ## keep.forest       logical    -  TRUE        -   -   FALSE     -
    ## keep.inbag        logical    - FALSE        -   -   FALSE     -

``` r
rforest_params <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("ntree", lower = 100, upper = 1000),
  # number of splits in each tree
  makeIntegerParam("mtry", lower = 1, upper = 10)
)

control <- makeTuneControlRandom(maxit = 1)

# Resampling plan
resample_desc <- makeResampleDesc("CV", iters = 4)

tuned_params <- tuneParams(
  learner = rforest_learner,
  task = trainTask,
  resampling = resample_desc,
  par.set = rforest_params,
  control = control
)

# Create a new model using tuned hyperparameters
rforest_tuned_learner <- setHyperPars(
  learner = rforest_learner,
  par.vals = tuned_params$x
)

# Make a new prediction
perf = list()
for(i in 1:5){
  set.seed(i)
  rforest_model <- train(rforest_tuned_learner, task = trainTask)
  predictions <- predict(rforest_model, testTask)
  perf[i] <- list(performance(predictions, measures = list(ppv, acc)))
}

mean(as.numeric(unname(as.data.table(perf)[1,])))
```

    ## [1] 0.7472734

``` r
mean(as.numeric(unname(as.data.table(perf)[2,])))
```

    ## [1] 0.8212291

``` r
rforest_tuned_results = c(mean(as.numeric(unname(as.data.table(perf)[1,]))), mean(as.numeric(unname(as.data.table(perf)[2,]))))

# KAGGLE
# retrain with full dataset
rforest_model <- train(rforest_tuned_learner, final_train)
final_predictions <- predict(rforest_model, final_test)
# prediction file
rforest_prediction <- final_predictions$data %>%
  select(PassengerID = id, Survived = response) %>%
  mutate(PassengerID = test_raw$PassengerId)

rforest_prediction$Survived = as.integer(rforest_prediction$Survived)-1

write.csv(rforest_prediction,"rforest_tuned_prediction.csv", row.names=F)
```

Parameter tuning also for xgboost classififer.

``` r
# To see all the parameters of the xgboost classifier
getParamSet("classif.xgboost")
```

    ##                                 Type  len             Def
    ## booster                     discrete    -          gbtree
    ## watchlist                    untyped    -          <NULL>
    ## eta                          numeric    -             0.3
    ## gamma                        numeric    -               0
    ## max_depth                    integer    -               6
    ## min_child_weight             numeric    -               1
    ## subsample                    numeric    -               1
    ## colsample_bytree             numeric    -               1
    ## colsample_bylevel            numeric    -               1
    ## num_parallel_tree            integer    -               1
    ## lambda                       numeric    -               1
    ## lambda_bias                  numeric    -               0
    ## alpha                        numeric    -               0
    ## objective                    untyped    - binary:logistic
    ## eval_metric                  untyped    -           error
    ## base_score                   numeric    -             0.5
    ## max_delta_step               numeric    -               0
    ## missing                      numeric    -                
    ## monotone_constraints   integervector <NA>               0
    ## tweedie_variance_power       numeric    -             1.5
    ## nthread                      integer    -               -
    ## nrounds                      integer    -               1
    ## feval                        untyped    -          <NULL>
    ## verbose                      integer    -               1
    ## print_every_n                integer    -               1
    ## early_stopping_rounds        integer    -          <NULL>
    ## maximize                     logical    -          <NULL>
    ## sample_type                 discrete    -         uniform
    ## normalize_type              discrete    -            tree
    ## rate_drop                    numeric    -               0
    ## skip_drop                    numeric    -               0
    ## callbacks                    untyped    -                
    ##                                      Constr Req Tunable Trafo
    ## booster                gbtree,gblinear,dart   -    TRUE     -
    ## watchlist                                 -   -   FALSE     -
    ## eta                                  0 to 1   -    TRUE     -
    ## gamma                              0 to Inf   -    TRUE     -
    ## max_depth                          1 to Inf   -    TRUE     -
    ## min_child_weight                   0 to Inf   -    TRUE     -
    ## subsample                            0 to 1   -    TRUE     -
    ## colsample_bytree                     0 to 1   -    TRUE     -
    ## colsample_bylevel                    0 to 1   -    TRUE     -
    ## num_parallel_tree                  1 to Inf   -    TRUE     -
    ## lambda                             0 to Inf   -    TRUE     -
    ## lambda_bias                        0 to Inf   -    TRUE     -
    ## alpha                              0 to Inf   -    TRUE     -
    ## objective                                 -   -   FALSE     -
    ## eval_metric                               -   -   FALSE     -
    ## base_score                      -Inf to Inf   -   FALSE     -
    ## max_delta_step                     0 to Inf   -    TRUE     -
    ## missing                         -Inf to Inf   -   FALSE     -
    ## monotone_constraints                -1 to 1   -    TRUE     -
    ## tweedie_variance_power               1 to 2   Y    TRUE     -
    ## nthread                            1 to Inf   -   FALSE     -
    ## nrounds                            1 to Inf   -    TRUE     -
    ## feval                                     -   -   FALSE     -
    ## verbose                              0 to 2   -   FALSE     -
    ## print_every_n                      1 to Inf   Y   FALSE     -
    ## early_stopping_rounds              1 to Inf   -   FALSE     -
    ## maximize                                  -   -   FALSE     -
    ## sample_type                uniform,weighted   Y    TRUE     -
    ## normalize_type                  tree,forest   Y    TRUE     -
    ## rate_drop                            0 to 1   Y    TRUE     -
    ## skip_drop                            0 to 1   Y    TRUE     -
    ## callbacks                                 -   -   FALSE     -

``` r
# We followed the standard parameter tuning procedure as shown in 
# the tutorial site of the mlr package
xgb_params <- makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("nrounds", lower = 100, upper = 500),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 1, upper = 10),
  # "shrinkage": counteracts overfitting
  makeNumericParam("eta", lower = .1, upper = .5),
  # L2 regularization also counteracts overfitting
  makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
)

control <- makeTuneControlRandom(maxit = 1)

# resampling
resample_desc <- makeResampleDesc("CV", iters = 4)

tuned_params <- tuneParams(
  learner = xgb_learner,
  task = trainTask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control
)

# Create a new model using tuned hyperparameters
xgb_tuned_learner <- setHyperPars(
  learner = xgb_learner,
  par.vals = tuned_params$x
)

# Make new predictions over 5 runs
perf = list()
for(i in 1:5){
  set.seed(i)
  xgb_model <- train(xgb_tuned_learner, task = trainTask)
  predictions <- predict(xgb_model, testTask)
  perf[i] <- list(performance(predictions, measures = list(ppv, acc)))
}

mean(as.numeric(unname(as.data.table(perf)[1,])))
```

    ## [1] 0.6388889

``` r
mean(as.numeric(unname(as.data.table(perf)[2,])))
```

    ## [1] 0.7821229

``` r
xcbg_tuned_results = c(mean(as.numeric(unname(as.data.table(perf)[1,]))), mean(as.numeric(unname(as.data.table(perf)[2,]))))

# KAGGLE
# Re-train parameters using tuned hyperparameters (and full training set)
xgb_model <- train(xgb_tuned_learner, final_train)
final_predictions <- predict(xgb_model, final_test)

# prediction file
xgboost_prediction <- final_predictions$data %>%
  select(PassengerID = id, Survived = response) %>%
  mutate(PassengerID = test_raw$PassengerId)

xgboost_prediction$Survived = as.integer(xgboost_prediction$Survived)-1
write.csv(xgboost_prediction,"xgboost_tuned_prediction.csv", row.names=F)
```

After submitting, we found the tuned RandomForest to perform best. On out evaluation setup, the following is obsoverd:

``` r
logreg_model <- train(logreg_learner, task = trainTask)
```

    ## Warning: glm.fit: algorithm did not converge

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
LOGREG <- performance(predict(logreg_model, testTask),measures = list(ppv, acc))
```

    ## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
    ## ifelse(type == : prediction from a rank-deficient fit may be misleading

``` r
rpart_model <- train(rpart_learner, task = trainTask)
RPART <- performance(predict(rpart_model, testTask),measures = list(ppv, acc))
results = data.frame()

results = rbind(LOGREG, RPART, rforest_results, xgb_results, xcbg_tuned_results, rforest_tuned_results)
results
```

    ##                             ppv       acc
    ## LOGREG                0.6285714 0.7709497
    ## RPART                 0.7368421 0.8212291
    ## rforest_results       0.7464629 0.8223464
    ## xgb_results           0.7333333 0.8268156
    ## xcbg_tuned_results    0.6388889 0.7821229
    ## rforest_tuned_results 0.7472734 0.8212291

Interestingly, the XGBoost algorithm worsens in performance after tuning.
