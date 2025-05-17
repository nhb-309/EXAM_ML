filepath <- "C:/Users/A829196/Downloads/bank/bank.csv"

data <- read.table(filepath, sep = ";", header = TRUE, stringsAsFactors = T)
#data <- data.frame(read.csv2(filepath))

###### Chargement des packages ######

library(skimr)
library(dplyr)
library(glmnet)
library(rpart)
library(randomForest)
library(pROC)
library(xgboost)
library(caret)
library(e1071)

###### Un peu de visualisation pour commencer ######

head(data)

summary(data)
skim(data)

sum(is.na(data))
#data <- na.omit(data)

# numerics <- which(sapply(data, is.numeric))
# factors <- colnames(data[,-numerics])
# 
# for(col in factors){
#   data[,col] <- as.factor(data[,col])
# }

###### Pré-traitement des données ######

outputVar <- "y"
colnames(data)[colnames(data)==outputVar] <- "Y"

prop.table(summary(data$Y))

# Retrait des colonnes inutiles, colonnes avec trop de NA, etc

# data <- data %>%
#   mutate(Y = as.factor(if_else(Y == "yes", 1, 0)))

###### Validation croisée ######

set.seed(1234)

nb=10


blocs <- sample(rep(1:nb, length=nrow(data)))
RES <- data.frame(Y=data$Y)

for(i in 1:nb){
  
  dataA <- data[blocs!=i,]
  dataT <- data[blocs==i,]
  
  ### Regression logistique ###
  regLog <- glm(Y~., data=dataA, family="binomial")
  RES[blocs==i, "RegLogis."] <- predict(regLog, dataT, type="response")
  
  ### Choix de variable avec STEP - AIC ###
  stepAIC <- stats::step(regLog, trace=0)
  RES[blocs==i, "AIC"] <- predict(stepAIC, dataT, type="response")
  
  ### Choix de variable avec STEP - BIC ###
  stepBIC <- stats::step(regLog, trace=0, k=log(nrow(dataT)))
  RES[blocs==i, "BIC"] <- predict(stepBIC, dataT, type="response")
  
  ### Preparation de la regularisation  ###
  XA <- model.matrix(Y~., data=dataA)[,-1]
  XT <- model.matrix(Y~., data=dataT)[,-1]
  
  YA <- as.matrix(dataA$Y)
  YT <- as.matrix(dataT$Y)
  
  ### Ridge ###
  ridge <- cv.glmnet(XA, YA, alpha=0, family="binomial")
  RES[blocs==i,"ridgeMin"] <- predict(ridge, XT, s="lambda.min", type="response")
  RES[blocs==i,"ridge1se"] <- predict(ridge, XT, s="lambda.1se", type="response")
  
  ### Lasso ###
  lasso <- cv.glmnet(XA, YA, alpha=1, family="binomial")
  RES[blocs==i,"lassoMin"] <- predict(lasso, XT, s="lambda.min", type="response")
  RES[blocs==i,"lasso1se"] <- predict(lasso, XT, s="lambda.1se", type="response")
  
  ### Elastic Net ###
  elNet <- cv.glmnet(XA, YA, alpha=0.5, family="binomial")
  RES[blocs==i,"elNetMin"] <- predict(elNet, XT, s="lambda.min", type="response")
  RES[blocs==i,"elNet1se"] <- predict(elNet, XT, s="lambda.1se", type="response")
  
  ### Arbre CART ###
  arbre <- rpart(Y ~ ., data = dataA, method = "class")  
  RES[blocs==i, "arbre"] <- predict(arbre, dataT, type = "prob")[, 2]
  
  # ### Random Forest ###
  # to do implémenter l'hyper-paramétrisation de ntrees
  # je crois que ntrees n' pas besoin d'etre parametre, il faut prendre un grand ntrees de base, exemple 1000
  rf <- randomForest(Y ~ ., data = dataA, ntree=500, type = "classification")
  RES[blocs == i, "foret"] <- predict(rf, dataT, type = "prob")[, 2]
  
  # ### XGBoost ###
  
  train_x <- data.matrix(subset(dataA, select=-c(Y)))
  test_x <-  data.matrix(subset(dataT, select=-c(Y)))
  
  
  # XGBoost sans recherche des hyper-paramètres
  #train_y <- as.numeric(dataA$Y)-1
  #test_y <- as.numeric(dataT$Y)-1
  #xgb_dataA <- xgb.DMatrix(data = train_x, label = train_y)
  #xgb_dataT <- xgb.DMatrix(data = test_x, label = test_y)
  # params <- list(
  #    objective = "binary:logistic",  
  #    eta = 0.1,                       # Taux d'apprentissage
  #    max_depth = 6,                   # Profondeur des arbres
  #    colsample_bytree = 0.8,          # Fraction des colonnes utilisées pour chaque arbre
  #    subsample = 0.8,                 # Fraction des lignes utilisées pour chaque arbre
  #    nrounds = 100                    # Nombre d'arbres (itérations)
  # )
  #  
  # xgb_mod <- xgboost(params = params, data = xgb_dataA, nrounds = params$nrounds, verbose = 0)
  # RES[blocs == i, "XGBoost"] <- predict(xgb_mod, xgb_dataT)
  
  # J'utilise `caret` pour la recherche des hyperparamètres de xgb
  train_y <- dataA$Y
  test_y <- dataT$Y
  
  tune_grid <- expand.grid(
    nrounds = c(50, 100, 150),               # Nombre d'arbres
    eta = c(0.01, 0.1, 0.2),                 # Taux d'apprentissage
    max_depth = c(3, 6, 10),                 # Profondeur des arbres
    colsample_bytree = c(0.6, 0.8, 1),       # Fraction des colonnes utilisées pour chaque arbre
    subsample = c(0.6, 0.8, 1),              # Fraction des lignes utilisées pour chaque arbre
    gamma = c(0, 0.1, 0.2),                   # Pénalité sur les arbres
    min_child_weight = c(1)
  )
  
  xgb_train_control <- trainControl(
    method = "cv",                            
    number = 5,                               
    verboseIter = FALSE,                      
    summaryFunction = twoClassSummary,        # métriques de classification binaire
    classProbs = TRUE                         # probabilités de classe
  )
  
  xgb_model <- train(
    x = train_x, 
    y = as.factor(train_y), 
    method = "xgbTree", 
    trControl = xgb_train_control, 
    tuneGrid = tune_grid,
    metric = "ROC"                          
  )
  
  best_params <- xgb_model$bestTune
  
  xgb_dataA <- xgb.DMatrix(data = train_x, label = as.numeric(train_y)-1)
  xgb_dataT <- xgb.DMatrix(data = test_x, label = as.numeric(test_y)-1)
  
  final_model <- xgboost(
    data = xgb_dataA,
    objective = "binary:logistic",
    nrounds = best_params$nrounds,
    eta = best_params$eta,
    max_depth = best_params$max_depth,
    colsample_bytree = best_params$colsample_bytree,
    subsample = best_params$subsample,
    gamma = best_params$gamma,
    verbose = 0
  )
  
  RES[blocs == i, "XGBoostGrid"] <- predict(final_model, xgb_dataT)
  
  #Je cherche si les params sont en bord de grille 
  grid <- xgb_model$results
  
  check_bord <- function(param_name) {
    best_val <- best_params[[param_name]]
    grid_vals <- unique(tune_grid[[param_name]])
    is_border <- best_val == min(grid_vals) || best_val == max(grid_vals)
    if (is_border) {
      paste(param_name, "est sur une bordure :", best_val)
    }
  }
  
  lapply(names(best_params), check_bord)
  
  ### SVM ###
  modSVMlin <- svm(Y~.,data=dataA, kernel="linear", probability = TRUE)
  RES[blocs==i, "SVMlin"] <- attr(predict(modSVMlin, dataT, probability = TRUE), "prob")[, 2]
  
  modSVMrad<- svm(Y~., data = dataA, type = "C-classification", kernel = "radial", probability = TRUE)
  RES[blocs==i, "SVMrad"] <- attr(predict(modSVMrad, dataT, probability = TRUE), "prob")[, 2]
  
}

###### Exploitation des résultats ######

rocAll <-roc(Y~., RES)

#Avec un seuil naturel pour une classification binaire (règle de Bayes)
metrics_seuil_05 <- sapply(rocAll, coords, x=0.5, ret=c("threshold", "accuracy", "sensitivity", "specificity"))
metrics_seuil_05 <- data.frame(metrics_seuil_05)

### !!! En cas de déséquilibre des données : F1-score à privilégier
calcul_F1 <- function(sensitivity, specificity) {
  print(as.numeric(sensitivity))
  print(as.numeric(specificity))
  
  precision <- (as.numeric(sensitivity)*as.numeric(specificity)) / ((1 - as.numeric(sensitivity)) + as.numeric(sensitivity)*as.numeric(specificity))
  F1 <- 2 * (precision * as.numeric(sensitivity)) / (precision + as.numeric(sensitivity))
  return(F1)
}


F1_scores <- lapply(metrics_seuil_05, function(x) calcul_F1(x["sensitivity"], x["specificity"]))
metrics_seuil_05 <- rbind(metrics_seuil_05, F1_score = F1_scores)

#Avec un seuil "best"
metrics_seuil_best <- sapply(rocAll, coords, x="best", ret=c("threshold", "accuracy", "sensitivity", "specificity"))
metrics_seuil_best <- data.frame(metrics_seuil_best)

F1_scores_best <- lapply(metrics_seuil_best, function(x) calcul_F1(x["sensitivity"], x["specificity"]))
metrics_seuil_best <- rbind(metrics_seuil_best, F1_score = F1_scores_best)

#Ou avec un seuil "maison" d'après la proportion de 1 dans Y : prop.table(summary(data$Y))



###### Un peu de feature engineering ######

# Optionnel : modele avec interactions
# Potentiellement n'ajouter les interactions que entre les var numériques ?
data <- data.frame(model.matrix(Y~.^2, data=data)[,-1], Y=data$Y)

# Optionnel : modele avec polynomes
tmp <- model.matrix(Y~., data=data)[,-1]
data <- data.frame(Y=data$Y, tmp, tmp^2, tmp^3)

# Potentiellement n'ajouter les polynomes que entre les var numériques ?
data <- data.frame(Y=data$Y, tmp, tmp[,numerics]^2, tmp[,numerics]^3)



# Ensuite ré-estimer sur l'ensemble du jeu de données et faire sur le jeu de test
# Puis faire du feature engineering : splines etc

# Question sur les groupes : si on a plusieurs groupes alors faire un modèle par groupe. ensuite comment
# gérer les nouvelles prédicions ? (il faut savoir dans quel groupe est l'individu). 
# on peut tenter une approche kmeans ? calculer la distance avec le centroïde de chaque groupe pour classer l'individu
