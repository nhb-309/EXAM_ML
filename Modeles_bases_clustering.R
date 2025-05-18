# Installation des packages ----
library(tidyverse)
library(gbm)
library(randomForest)
library(dbscan)
library(caret)
library(tidymodels)
library(recipes)
library(xgboost)
library(glmnet)
library(splines)

setwd('L:/REVISION/')
list.files()


setwd("L:/REVBDF")


data=read.csv('crop.csv', sep=',') %>% select(-label)

## Visualisation rapide----
data %>% 
    summary()

data %>% str()

## Coder la variable cible en y ----
data <- data %>% 
    rename(Y=ph)

## Traitement des valeurs manquantes ----

# SI PEU DE VALEURS manquantes, RETRAIT DES NA
data <- data %>% 
    na.exclude()

# SI TROP DE VALEURS MANQUANTES, RETRAIT VARIABLE

# Imputation des valeurs manquantes : on fera cette étape pour le jeu train/test séparemment

## Recoder les facteurs numériques en facteurs ----
data <- data %>% 
    mutate(across(.cols=where(is.character), .fns = as.factor))


# TRAVAIL DES JEUX TRAIN VS TEST ----

# reproductibilité
set.seed(1234)

dataset_split <- initial_split(data,prop = 0.9,strata = Y)

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


# ==============================================================================
# >>> Clustering 
# Classification ascendante hiérarchique
# ==============================================================================

# Matrice de distances
mat_dist = dist(don,method='euclidean')

# Calcul de la CAH
cah=hclust(mat_dist, method= 'ward.D')

# Identifier le point de rupture 
plot(rev(cah$height)[1:10], type='b')
plot(cah)
n.clusters = 3
don$cluster = cutree(cah, k=n.clusters) # définir le nombre de clusters


# ==============================================================================
# >>> Boucler sur n.clusters 
# Classification ascendante hiérarchique
# ==============================================================================



# BOUCLE DE COMPARAISON ----
jj=1;ii=1
for(jj in 1:n.clusters){
    
    don_clust = don[which(don$cluster == jj),]
    
    don_clustX = as.matrix(don_clust %>% 
                         select(!Y))
    
    don_clustY = don_clust$Y
    
    
    SCORE=data.frame('Y'=don_clust$Y)
    
    nb=3
    
    bloc <- sample(rep(1:nb,length=nrow(don_clust)))  # création des blocs de la VC
    
    for(ii in 1:nb){
    
    
    t1 <- Sys.time()
    
    print(ii)
    
    print("Séparation blocs train/test")
    # Apprentissage v test
    donA <- don_clust[bloc!=ii,]
    donT <- don_clust[bloc==ii,]
    
    # Apprentissage v test matrice
    donXA <- donX[bloc!=ii,]
    donXT <- donX[bloc==ii,]
    donYA <- donY[bloc!=ii]
    
    print("Régression Linéaire")
    ## Linéaire =================================================================
    tmp <- lm(Y~.,data=donA)
    SCORE[bloc==ii,"lm"] <- predict(tmp,donT)
    
    print("STEP AIC")
    # AIC ========================================================================
    #tmp_aic <- stats::step(tmp,trace=0)
    #SCORE[bloc==ii,"aic"] <- predict(tmp_aic,donT)
    #
    #print("STEP BIC")
    ## BIC ========================================================================
    #tmp_bic <- stats::step(tmp,trace=0,k=log(nrow(donA)))
    #SCORE[bloc==ii,"bic"] <- predict(tmp_bic,donT)
    #
    ## Penalisation =================================================================
    #print("Ridge")
    #ridge=cv.glmnet(donXA,donYA,alpha=0) 
    #
    #print("Lasso")
    #lasso=cv.glmnet(donXA,donYA,alpha=1)
    #
    #print("Elastic Net")
    #elnet=cv.glmnet(donXA,donYA,alpha=0.5)
    #
    #SCORE[bloc==ii,'ridge']=predict(ridge,newx=donXT,s='lambda.min')
    #SCORE[bloc==ii,"ridge1se"] <- predict(ridge,donXT,s="lambda.1se")
    #
    #SCORE[bloc==ii,'lasso']=predict(lasso,newx=donXT,s='lambda.min')
    #SCORE[bloc==ii,"lasso1se"] <- predict(lasso,donXT,s="lambda.1se")
    #
    #SCORE[bloc==ii,'elnet']=predict(elnet,newx=donXT,s='lambda.min')
    #SCORE[bloc==ii,"elas1se"] <- predict(elnet,donXT,s="lambda.1se")
    #
    #print("Forêt")
    ## Forêt ========================================================================
    #
    #### Pas d'hypoer-parametrage ----
    #
    #foret <- randomForest(Y~.,data=donA)
    #SCORE[bloc==ii,"foret"] <- predict(foret,donT)
    #
    #
    #### Hyper-parametrage ----
    #
    #print("hyprparamètrage de la forêt")
    #
    #results <- data.frame()
    #
    #for (j in seq(1:nrow(gr.foret))) {
    #    
    #    tuned_model <- randomForest(Y~.,
    #                                data=donA,
    #                                ntree=gr.foret$num.trees[j],
    #                                mtry = gr.foret$mtry[j],
    #                                nodesize= gr.foret$nodesize[j])
    #    
    #    # Calculer l'erreur quadratique moyenne (RMSE)
    #    predictions <- predict(tuned_model, donT)
    #    rmse <- sqrt(mean((predictions - donT$Y)^2))
    #    
    #    results <- rbind(results, data.frame(ntree = gr.foret$num.trees[j],
    #                                         mtry = gr.foret$mtry[j],
    #                                         nodesize= gr.foret$nodesize[j],
    #                                         RMSE = rmse))
    #}
    #
    #
    #### Selection du meilleur parametrage
    #
    #best.foret.params <-  results[which.min(results$RMSE), ]
    #
    #best.foret = randomForest(Y~.,
    #                          data=donA,
    #                          ntree=best.foret.params$ntree,
    #                          mtry = best.foret.params$mtry,
    #                          nodesize= best.foret.params$nodesize)
    #
    #
    #SCORE[bloc==ii, 'best foret']= predict(best.foret, donT)
    #
    #
    #print("gbm")
    #
    ## gbm  =============================
    #
    #### sans hyperparamétrage ----
    #gb <- gbm(Y~.,data = donA,
    #          distribution = "gaussian",
    #          cv.folds = 5)
    #
    #SCORE[bloc==ii,"gbm tot"] <- predict(gb,donT)
    #
    #### avec hyperparamétrage ----
    #
    #tmp=gbm(Y~.,
    #        data=donA,
    #        distribution = 'gaussian',
    #        cv.fold=5,
    #        n.trees=300,
    #        interaction.depth=3,
    #        shrinkage=0.5)
    #
    #nopt=gbm.perf(tmp,method='cv') # permet d'évaluer la perf du boost. vert=erreur de prev / noir=erreur d'estim.
    #
    #gbopt <- gbm(Y~.,data = donA,distribution = "gaussian",n.trees = nopt)
    #
    #SCORE[bloc==ii,"gbm opt"] <- predict(gbopt,donT)
    #
    #print("XGBOOST")
    #
    ## XG Boost optimisé ====================
    #
    #xgb_data_train=xgb.DMatrix(data = donXA, label = donYA)
    #xgb_data_test=xgb.DMatrix(data = donXT)
    #
    #results = data.frame()
    #
    #
    ## AJOUTER UN BORD DE GRILLE
    #for(pp in 1:nrow(xgrid)){
    #    
    #    print(paste0("test de la grille ",pp,"/",nrow(xgrid)))
    #    
    #    params <- list(objective = "reg:squarederror",
    #                   max_depth = xgrid$max_depth[pp],
    #                   eta=xgrid$eta[pp],
    #                   eval_metric="rmse")
    #    
    #    nIter = 300
    #    
    #    tmp = xgb.cv(params = params,
    #                 data=xgb_data_train,
    #                 nfold=5,
    #                 nrounds = nIter,
    #                 early_stopping_rounds = 10,
    #                 verbose = F)
    #    
    #    bestIter = tmp$best_iteration
    #    
    #    if(bestIter==nIter){
    #        print("ATTENTION: BORD DE GRILLE")
    #    }
    #    
    #    bestRMSE = tmp$evaluation_log$test_rmse_mean[bestIter]
    #    
    #    x = data.frame(cbind(xgrid[pp,],bestIter,bestRMSE))
    #    
    #    results[pp,1:nrow(xgrid)] = x
    #    
    #}
    #
    #results
    #
    #best_params_index <- results[which.min(results$bestRMSE),] 
    #
    #best_params = list(
    #    objective = "reg:squarederror",
    #    eval_metric = "rmse",
    #    max_depth = best_params_index$max_depth,
    #    eta = best_params_index$eta
    #)
    #
    #xgbopt = xgb.train(params = best_params,
    #                   data = xgb_data_train,
    #                   nrounds = best_params_index$bestIter)
    #
    #SCORE[bloc==ii,"xgb"] <- predict(xgbopt,xgb_data_test)
    #
    print(paste0("Temps total du bloc",ii,": ",(Sys.time()-t1) %>% round(2)))
    }
    
    SCORE$cluster = jj
    
    assign(paste0('SCORE_', 'cluster_',jj ),SCORE)
    
}

tobind=ls(pattern = 'SCORE_cluster',envir = .GlobalEnv)
score_list=mget(tobind)
SCORE_clusters =  do.call(rbind, Map(function(x, name) {
    x$source <- name
    x
}, score_list, names(score_list)))

rownames(SCORE_clusters)=NULL

SCORE_clusters %>% head()

# ESTIMATION SUR LE JEU D'APPRENTISSAGE ----

SCORE
erreur2 = function(X,Y){mean((X-Y)^2)}
sort((apply(SCORE,2,erreur2,Y=SCORE[,'Y'])[-1]))







