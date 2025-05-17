# SPLINES ----

# Identification des variables numériques 

## Variables catégorielles
colCat <-colnames(data)[grepl('factor|logical|character',sapply(data, class))]

## retrait de la variable Y
colCat<-colCat[colCat!="Y"]

## df de variables cat
dataCat <- data[,colCat]

## Variables numériques
colNum <- colnames(data)[grepl('numeric|integer',sapply(data,class))]

## retrait de la variable Y
colNum<-colNum[colNum!="Y"]

# df de variables num
dataNum <- data[,colNum]

#On fait des splines de degré 3 avec 3 noeuds situés aux quantiles 
# 0.25, 0.5 et 0.75 de chaque variable
# Si besoin : rec_spline <- rec_interaction %>% step_ns(age)

data_splines <- data

for(ii in 1:ncol(dataNum)){
  
  var <- dataNum[,ii]
  
  degree = 3
  
  knots = quantile(var, probs = c(0.25, 0.5, 0.75))
  
  BB <- bs(var, degree = degree, knots = knots)
  
  colnames(BB) <- paste(colnames(dataNum)[ii], 1:(degree+length(knots)),sep = "_")
  
  #6 = degree-1 x nombre de noeux
  data_splines <- cbind(data_splines,BB)
}

# TRAVAIL DES JEUX TRAIN VS TEST ----

# reproductibilité
set.seed(1234)

## Séparation en train et test ----

# Séparation façon classique

# split=0.9
# 
# trainSize=round(split*nrow(data))
# 
# A = sample(nrow(data), trainSize)
# 
# don  = data[A,]
# 
# donnees_test_final <- data[-A,]

# Séparation façon tidy

dataset_split <- initial_split(data_splines,
                               prop = 0.55, # on réduit le nb d'observations pour que ça tourne
                               strata = Y)

don <- training(dataset_split)

donnees_test_final <- testing(dataset_split)

Y_FIN <- donnees_test_final$Y

donnees_test_final <- donnees_test_final %>% 
  select(!Y)

## Jeu d'entrainement ----

### Recettes ----

my_recipe = recipe(Y ~ ., data = don)  %>% 
  # Retrait des variables avec une seule occurence
  step_zv(all_predictors())%>% 
  # Imputer les valeurs manquantes numériques la moyenne
  # step_impute_mean(all_numeric_predictors()) %>% 
  # Normaliser les variables numétiques
  step_normalize(all_numeric_predictors()) %>%
  # # Imputer aux valeurs manquantes cat le mode
  # step_impute_mode(all_nominal_predictors()) %>%
  # # Imputer la valeur 'autres' aux variables avec trop d'occurence
  # step_other(dp,threshold = 0.03) %>% 
  # One hot encoding
  step_dummy(all_nominal_predictors(),one_hot = TRUE)


don <- my_recipe %>% 
  prep() %>% 
  bake(new_data = don)

### Création du format matriciel ---- 
donX = as.matrix(don %>% 
                   select(!Y))

donY = don$Y

## Jeu de test----

### Recettes ----

my_recipe = recipe(~ ., data = donnees_test_final)  %>% 
  # Retrait des variables avec une seule occurence
  step_zv(all_predictors())%>% 
  # Imputer les valeurs manquantes numériques la moyenne
  # step_impute_mean(all_numeric_predictors()) %>% 
  # Normaliser les variables numétiques
  step_normalize(all_numeric_predictors()) %>%
  # # Imputer aux valeurs manquantes cat le mode
  # step_impute_mode(all_nominal_predictors()) %>%
  # # Imputer la valeur 'autres' aux variables avec trop d'occurence
  # step_other(dp,threshold = 0.03) %>% 
  # One hot encoding
  step_dummy(all_nominal_predictors(),one_hot = TRUE)

donnees_test_final <- my_recipe %>% 
  prep() %>% 
  bake(new_data = donnees_test_final)

### Création du format matriciel ----

donnees_test_finalX = as.matrix(donnees_test_final)

donnes_test_finalY = Y_FIN




# PREPA COMPARAISON D'ALGORITHME ----

# reproductibilité
set.seed(1234)

## division en blocs pour cross validation ----
nb <- 3

bloc <- sample(rep(1:nb,length=nrow(don)))

bloc

table(bloc)

## Grilles ----

### Foret ----
gr.foret=expand.grid(num.trees=c(100,500),
                     mtry=c(3,5,7),
                     nodesize = c(5,10)) %>% 
  data.frame()

### XGBOOST ----
xgrid = expand.grid(
  max_depth = c(1,2),
  eta = c(0.1,0.05)
)

## définition de la table de résultats finaux ----

SCORE_splines=data.frame('Y'=don$Y)

# BOUCLE DE COMPARAISON ----

for(ii in 1:nb){
  
  
  t1 <- Sys.time()
  
  print(ii)
  
  print("Séparation blocs train/test")
  # Apprentissage v test
  donA <- don[bloc!=ii,]
  donT <- don[bloc==ii,]
  
  # Apprentissage v test matrice
  donXA <- donX[bloc!=ii,]
  donXT <- donX[bloc==ii,]
  donYA <- donY[bloc!=ii]
  
  print("Régression Linéaire")
  ## Linéaire =================================================================
  tmp <- lm(Y~.,data=donA)
  SCORE_splines[bloc==ii,"lm"] <- predict(tmp,donT)
  
  print("STEP AIC")
  # AIC ========================================================================
  tmp_aic <- stats::step(tmp,trace=0)
  SCORE_splines[bloc==ii,"aic"] <- predict(tmp_aic,donT)
  
  print("STEP BIC")
  # BIC ========================================================================
  tmp_bic <- stats::step(tmp,trace=0,k=log(nrow(donA)))
  SCORE_splines[bloc==ii,"bic"] <- predict(tmp_bic,donT)
  
  # Penalisation =================================================================
  print("Ridge")
  ridge=cv.glmnet(donXA,donYA,alpha=0) 
  
  print("Lasso")
  lasso=cv.glmnet(donXA,donYA,alpha=1)
  
  print("Elastic Net")
  elnet=cv.glmnet(donXA,donYA,alpha=0.5)
  
  SCORE_splines[bloc==ii,'ridge']=predict(ridge,newx=donXT,s='lambda.min')
  SCORE_splines[bloc==ii,"ridge1se"] <- predict(ridge,donXT,s="lambda.1se")
  
  SCORE_splines[bloc==ii,'lasso']=predict(lasso,newx=donXT,s='lambda.min')
  SCORE_splines[bloc==ii,"lasso1se"] <- predict(lasso,donXT,s="lambda.1se")
  
  SCORE_splines[bloc==ii,'elnet']=predict(elnet,newx=donXT,s='lambda.min')
  SCORE_splines[bloc==ii,"elas1se"] <- predict(elnet,donXT,s="lambda.1se")
  
  print("Forêt")
  # Forêt ========================================================================
  
  ### Pas d'hypoer-parametrage ----
  
  foret <- randomForest(Y~.,data=donA)
  SCORE_splines[bloc==ii,"foret"] <- predict(foret,donT)
  
  
  ### Hyper-parametrage ----
  
  print("hyprparamètrage de la forêt")
  
  results <- data.frame()
  
  for (j in seq(1:nrow(gr.foret))) {
    
    tuned_model <- randomForest(Y~.,
                                data=donA,
                                ntree=gr.foret$num.trees[j],
                                mtry = gr.foret$mtry[j],
                                nodesize= gr.foret$nodesize[j])
    
    # Calculer l'erreur quadratique moyenne (RMSE)
    predictions <- predict(tuned_model, donT)
    rmse <- sqrt(mean((predictions - donT$Y)^2))
    
    results <- rbind(results, data.frame(ntree = gr.foret$num.trees[j],
                                         mtry = gr.foret$mtry[j],
                                         nodesize= gr.foret$nodesize[j],
                                         RMSE = rmse))
  }
  
  
  ### Selection du meilleur parametrage
  
  best.foret.params <-  results[which.min(results$RMSE), ]
  
  best.foret = randomForest(Y~.,
                            data=donA,
                            ntree=best.foret.params$ntree,
                            mtry = best.foret.params$mtry,
                            nodesize= best.foret.params$nodesize)
  
  
  SCORE_splines[bloc==ii, 'best foret']= predict(best.foret, donT)
  
  
  print("gbm")
  
  # gbm  =============================
  
  ### sans hyperparamétrage ----
  gb <- gbm(Y~.,data = donA,
            distribution = "gaussian",
            cv.folds = 5)
  
  SCORE_splines[bloc==ii,"gbm tot"] <- predict(gb,donT)
  
  ### avec hyperparamétrage ----
  
  tmp=gbm(Y~.,
          data=donA,
          distribution = 'gaussian',
          cv.fold=5,
          n.trees=300,
          interaction.depth=3,
          shrinkage=0.5)
  
  nopt=gbm.perf(tmp,method='cv') # permet d'évaluer la perf du boost. vert=erreur de prev / noir=erreur d'estim.
  
  gbopt <- gbm(Y~.,data = donA,distribution = "gaussian",n.trees = nopt)
  
  SCORE_splines[bloc==ii,"gbm opt"] <- predict(gbopt,donT)
  
  print("XGBOOST")
  
  # XG Boost optimisé ====================
  
  xgb_data_train=xgb.DMatrix(data = donXA, label = donYA)
  xgb_data_test=xgb.DMatrix(data = donXT)
  
  results = data.frame()
  
  
  # AJOUTER UN BORD DE GRILLE
  for(pp in 1:nrow(xgrid)){
    
    print(paste0("test de la grille ",pp,"/",nrow(xgrid)))
    
    params <- list(objective = "reg:squarederror",
                   max_depth = xgrid$max_depth[pp],
                   eta=xgrid$eta[pp],
                   eval_metric="rmse")
    
    nIter = 300
    
    tmp = xgb.cv(params = params,
                 data=xgb_data_train,
                 nfold=5,
                 nrounds = nIter,
                 early_stopping_rounds = 10,
                 verbose = F)
    
    bestIter = tmp$best_iteration
    
    if(bestIter==nIter){
      print("ATTENTION: BORD DE GRILLE")
    }
    
    bestRMSE = tmp$evaluation_log$test_rmse_mean[bestIter]
    
    x = data.frame(cbind(xgrid[pp,],bestIter,bestRMSE))
    
    results[pp,1:nrow(xgrid)] = x
    
  }
  
  results
  
  best_params_index <- results[which.min(results$bestRMSE),] 
  
  best_params = list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = best_params_index$max_depth,
    eta = best_params_index$eta
  )
  
  xgbopt = xgb.train(params = best_params,
                     data = xgb_data_train,
                     nrounds = best_params_index$bestIter)
  
  
  SCORE_splines[bloc==ii,"xgb"] <- predict(xgbopt,xgb_data_test)
  
  print(paste0("Temps total du bloc",ii,": ",(Sys.time()-t1) %>% round(2)))
}



SCORE_splines
erreur2 = function(X,Y){mean((X-Y)^2)}
sort((apply(SCORE_splines,2,erreur2,Y=SCORE_splines[,'Y'])[-1]))


# Estimation modèle total

# Choisir les donnees qui ont donné le


# xgboost a gagné ----

## Création des matrices train et tests finales


xgb_data_train_final=xgb.DMatrix(data = donX, label = donY)

xgb_data_test_final=xgb.DMatrix(data = donnees_test_finalX)

results = data.frame()


for(pp in 1:nrow(xgrid)){
  
  print(paste0("test de la grille ",pp,"/",nrow(xgrid)))
  
  params <- list(objective = "reg:squarederror",
                 max_depth = xgrid$max_depth[pp],
                 eta=xgrid$eta[pp],
                 eval_metric="rmse")
  
  nIter = 300
  
  tmp = xgb.cv(params = params,
               data=xgb_data_train_final,
               nfold=5,
               nrounds = nIter,
               early_stopping_rounds = 10,
               verbose = F)
  
  bestIter = tmp$best_iteration
  
  if(bestIter==nIter){
    print("ATTENTION: BORD DE GRILLE")
  }
  
  bestRMSE = tmp$evaluation_log$test_rmse_mean[bestIter]
  
  x = data.frame(cbind(xgrid[pp,],bestIter,bestRMSE))
  
  results[pp,1:nrow(xgrid)] = x
  
}

results

best_params_index <- results[which.min(results$bestRMSE),] 

best_params = list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = best_params_index$max_depth,
  eta = best_params_index$eta
)

xgbopt = xgb.train(params = best_params,
                   data = xgb_data_train_final,
                   nrounds = best_params_index$bestIter)


Y_pred_fin <- predict(xgbopt,xgb_data_test_final)

RES <- data.frame(Y_obs = Y_FIN,model_pred_splines= Y_pred_fin)

erreur2 = function(X,Y){mean((X-Y)^2)}

model_pred_splines <- (apply(X = RES,MARGIN = 2,FUN = erreur2,Y=RES[,'Y_obs']))[-1]

model_pred_splines

