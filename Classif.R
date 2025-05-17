

library(bestglm)
library(caret) # necessaire pour la validation croisee stratifiee et le cv
library(tidyverse)
library(pROC)
library(gbm)
library(ranger)
library(glmnet)
library(xgboost)
library(tidymodels)
library(recipes)
library(bestglm)
library(kernlab)
library(data.table)
ncores = parallel::detectCores()

# Répertoire de travail
setwd("F:/DataScientist/revisions_ML/")

db = read.csv('diab.csv',sep=',') %>% 
    filter(CLASS!='P') %>% 
    mutate(CLASS = case_when(CLASS == 'N'~0,
                             CLASS == 'Y'~1)) %>% 
    mutate(CLASS = as.factor(CLASS)) %>%
    rename(Y= CLASS) %>% 
    mutate(Gender = case_when(Gender == 'F'~0,
                              Gender == 'M'~1)) %>% 
    mutate(Gender = as.factor(Gender)) %>% 
    select(-c(ID,No_Pation))
table(db$Y)
table(db$Gender)
dim(db)


db = SAheart %>% 
    rename(Y=chd) %>% 
    mutate(Y=as.factor(Y)) %>% 
    mutate(famhist = case_when(famhist=='Present'~1,
                               famhist=='Absent'~0))

# lecture
db = read.csv('bank.csv',sep=';')  %>% 
    rename(Y=y) %>%
    mutate(Y = case_when(Y=='no'~0,
                         Y=='yes'~1)) %>% 
    mutate(Y=as.factor(Y))

# Retrait des NA, mais uniquement s'il y en a peu
db = db %>% na.exclude()
# strings as factors
db = db %>% mutate(across(.cols=where(is.character),.fns=as.factor))

# fonctions de résumé, informations sur le jeu de données
summary(db)
sapply(db, class)
boxplot(db)
db %>% str()


# Graine
set.seed(12)

# tidymodels pour split en 2 jeux de données TRAIN et TEST

dbsplit = initial_split(db, prop=0.9, strata=Y)

donApp = training(dbsplit)
donTest = testing(dbsplit)

donTest.Y = donTest$Y

donTest = donTest %>% select(-Y)

# Recettes avec tidymodels
recipe.APP = recipe(Y ~ ., data = donApp) %>% 
    step_zv(all_predictors()) %>% 
    step_normalize(all_numeric_predictors()) %>% 
    #step_impute_median(all_numeric_predictors()) %>% 
    #step_impute_mode(all_nominal_predictors()) %>% 
    step_dummy(all_nominal_predictors(),one_hot = T) 

recipe.TEST = recipe(~ ., data = donTest) %>% 
    step_zv(all_predictors()) %>% 
    step_normalize(all_numeric_predictors()) %>% 
    #step_impute_median(all_numeric_predictors()) %>% 
    #step_impute_mode(all_nominal_predictors()) %>% 
    step_dummy(all_nominal_predictors(),one_hot=T) 

# Prep / Bake
donApp = recipe.APP %>% prep() %>% bake(new_data=donApp)
donTest= recipe.TEST %>% prep() %>% bake(new_data=donTest)

# Matrices de modèle
donAppX = model.matrix(Y~., data=donApp)[,-1]
donAppY = donApp$Y

nbloc=3

folds = createFolds(donApp$Y,
                    k=nbloc,
                    list=F) # a voir si createFolds garde la proportion automatiquement



# ==============================================================================
# Grilles
# ==============================================================================

# Foret
gr.foret=expand.grid(num.trees=c(100,300),
                     max.depth=c(1,5))

gr.foret.params=data.frame(gr.foret,'auc'=NA)

# SVM
gr.poly=expand.grid(C=c(0.1,10,100),
                    degree=c(1,2,3),
                    scale=1)
gr.radial=expand.grid(C=c(0.1,1,10),
                      sigma = c(0.0001,0.001,0.01,0.1,1))

ctrl = trainControl(method='cv', number=5)


# Validation croisee
SCORE=data.frame('Y'=donApp$Y,'logistic'=NA,'aic'=NA,'bic'=NA,'ridge'=NA,'lasso'=NA,'elnet'=NA,'foret'=NA,'rad_svm'=NA,'pol_svm'=NA,'gbm'=NA,'xgb'=NA)

# ==============================================================================
# > Comparaison des modèles
# ==============================================================================
jj=1
for(jj in 1:nbloc){
  cat('Fold: ', jj, '\n')
  
  donA=donApp[folds!=jj,]
  donV=donApp[folds==jj,]
  
  donXA=donAppX[folds!=jj,]
  donXV=donAppX[folds==jj,]  
  donYA=donAppY[folds!=jj]
  
  
  # Logistique =================================================================
  logistic=glm(Y~., data=donA, family='binomial')
  SCORE[folds==jj,'logistic'] = predict(logistic,newdata=donV,type='response')
  
  # AIC ========================================================================
  #aic=stats::step(logistic,trace=0)
  #SCORE[folds==jj,'aic']=predict(aic,newdata=donV,type='response')
  
  ## BIC ========================================================================
  #bic=stats::step(logistic,trace=0,k=log(nrow(donA)))
  #SCORE[folds==jj,'bic']=predict(bic,newdata=donV,type='response')
  
  # Penalisation =================================================================
  
  ridge=cv.glmnet(donXA,donYA,alpha=0  ,family='binomial',nfolds=5,type.measure='auc')
  lasso=cv.glmnet(donXA,donYA,alpha=1  ,family='binomial',nfolds=5,type.measure='auc')
  elnet=cv.glmnet(donXA,donYA,alpha=0.5,family='binomial',nfolds=5,type.measure='auc')
  SCORE[folds==jj,'ridge']=predict(ridge,newx=donXV,type='response',s='lambda.min')
  SCORE[folds==jj,'lasso']=predict(lasso,newx=donXV,type='response',s='lambda.min')
  SCORE[folds==jj,'elnet']=predict(elnet,newx=donXV,type='response',s='lambda.min')
  
  # Foret ======================================================================
  
  ### Hyper-parametrage
  
  control <- trainControl(method="cv",number=5)
  
  best_params <- data.frame()
  
  for (j in 1:nrow(gr.foret)) {
      cat('Foret - ', j, '\n')
      tuned_model <- train(Y~.,data=donA,
                           method="ranger",
                           classification = T,
                           metric = 'Accuracy',
                           trControl=control,
                           tuneGrid=expand.grid(
                               mtry=c(1,3),
                               splitrule='gini',
                               min.node.size=c(1,3)),
                           num.trees=gr.foret$num.trees[j],
                           max.depth=gr.foret$max.depth[j])
      
      results = tuned_model$results
      
      params = results[which.max(results$Accuracy),] %>% 
          mutate(num.trees=gr.foret$num.trees[j],
                 max.depth=gr.foret$max.depth[j])
      
      best_params = rbind(best_params, params)
  }
  
  param_optimaux = best_params[which.max(best_params$Accuracy),]
  
  foret.finale=ranger(factor(Y)~., data=donA,
                      classification=T, probability = T,
                      num.trees=param_optimaux$num.trees,
                      max.depth = param_optimaux$max.depth,
                      min.bucket = param_optimaux$min.buckets,
                      mtry = param_optimaux$mtry)
  
  SCORE[folds==jj,'foret'] = predict(foret.finale,data=donV,type='response')$predictions[,'1']
  
  
  
  # SVM ========================================================================
  #svm.poly = train(Y~., data=donA, method = 'svmPoly', trControl = ctrl, tuneGrid = gr.poly)
  #bestPoly=svm.poly$results[which.max(svm.poly$results$Accuracy),]
  #tmpPol=ksvm(Y~., data=donA, kernel = 'polydot', kpar=list(degree=bestPoly$degree,scale=1,offset=1),C=bestPoly$C,prob.model=T)
  #
  #svm.radial = train(Y~., data=donA, method = 'svmRadial', trControl = ctrl, tuneGrid = gr.radial)
  #bestRadial=svm.radial$results[which.max(svm.radial$results$Accuracy),]
  #tmpRad=ksvm(Y~., data=donA, kernel =  'rbfdot', kpar=list(sigma=bestRadial$sigma),C=bestRadial$C,prob.model=T)
  #
  #SCORE[folds==jj,'pol_svm'] = predict(tmpPol, newdata=donV, type = 'prob')[,'1'] # rajouter l'index de colonne : il ne faut en prendre qu'une
  #SCORE[folds==jj,'rad_svm'] = predict(tmpRad, newdata=donV, type= 'prob')[,'1']  
  
 
  ## =============================================================================
  ## Gradient Boosting
  ## =============================================================================
  # 1. Prepare data: ensure binary response is 0/1
  donA.gbm = donA %>% mutate(Y = as.numeric(Y) - 1)
  
  tmp <- gbm(Y~.,data=donA.gbm,distribution = "bernoulli",cv.folds=5,n.trees=300,
             interaction.depth = 3,shrinkage = 0.1)
  best.iter <- gbm.perf(tmp, method = "cv")
  SCORE[folds==jj,"gbm"] = predict.gbm(tmp,donV,type='response',n.trees = best.iter)
  
  # XGBOOST =====================================================================
  
  indY = which(names(donA)=='Y')
  
  indY
  
  X_train = as.matrix(donA[,-indY])
  
  Y_train = as.numeric(donA[[indY]])-1
  
  X_test = as.matrix(donV[,-indY])
  
  Y_test = as.numeric(donV[[indY]])-1
  
  xgb_data_train=xgb.DMatrix(data = X_train, label = Y_train)
  
  xgb_data_test=xgb.DMatrix(data = X_test, label = Y_test)
  
  xgrid = expand.grid(
    max_depth = c(1,5),
    eta = c(0.1),
    lambda = c(0,10),
    alpha=c(0,0.5,1)
  )
  
  results = data.frame()
  
  

  for(pp in 1:nrow(xgrid)){
    
    cat('\n',100*pp/nrow(xgrid),'% \n')
    
    params = list(
      objective = "binary:logistic",
      metrics = "logloss",
      stratified = T,
      max_depth = xgrid$max_depth[pp],
      alpha = xgrid$alpha[pp],
      lambda = xgrid$lambda[pp],
      eta = xgrid$eta[pp]
    )
    
    nIter = 600
    
    tmp = xgb.cv(
      params = params,
      nrounds = nIter,
      nfold = 5,
      verbose = F,
      early_stopping_rounds = 10,
      data=xgb_data_train
    )
      
    bestIter = tmp$best_iteration
    
    #if(bestIter==nIter){
    #  cat('\n',"ATTENTION: BORD DE GRILLE: ", bestIter, '\n')
    #} else{
    #    cat('\n' ,'Meilleure itération XGBoost : ', bestIter, '\n' )
    #}
    
    bestLogloss = tmp$evaluation_log$test_logloss_mean[bestIter]
    
    x = data.frame(cbind(xgrid[pp,],bestIter,bestLogloss))
    
    results[pp,1:nrow(xgrid)] = x
  }
  
  best_params_index <- results[which.max(results$bestLogloss),] 
  
  checkIter = best_params_index$bestIter
  
  if(checkIter == nIter){
      cat('\n', "ATTENTION: BORD DE GRILLE --> ", checkIter,'\n')
  }else{
      cat('\n','Meilleure itération XGBoost: ', checkIter,'\n')
  }
  
  best_params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = best_params_index$max_depth,
    eta = best_params_index$eta,
    alpha = best_params_index$alpha,
    lambda = best_params_index$lambda
  )
  
  xgb = xgb.train(params = best_params,
                  data = xgb_data_train, 
                  nrounds = best_params_index$bestIter)
  
  SCORE[folds==jj, 'xgb']=  predict(xgb,newdata = xgb_data_test,type='response')
  
}

# ==============================================================================
# > Comparaison des modèles
# ==============================================================================

SCORE

rocCV = roc(factor(Y)~., data=SCORE %>% select(-c(aic,bic,rad_svm,pol_svm)), quiet=T)

aucmodele = sort(round(unlist(lapply(rocCV,auc)),5),dec=TRUE)

tmp = lapply(rocCV, FUN = function(r) {
    co = coords(r, x = "best",
                ret = c('threshold','tp','fp','fn','tn','sensitivity','specificity','accuracy'),
                transpose = TRUE)
    
    # Compute F1 score
    tp = co["tp"]
    fp = co["fp"]
    fn = co["fn"]
    f1 = if ((2 * tp + fp + fn) == 0) NA else 2 * tp / (2 * tp + fp + fn)
    
    # Add F1 to the coords result
    co["f1"] <- f1
    return(co)
})
mat=do.call(rbind,tmp)

aucmodele
mat

# ==============================================================================
# > Entraînement du meilleur modèle. 
# ==============================================================================

#> repartir du jeu d'apprentissage complet
#> optimiser les hyperparamètres éventuels de l'algorithme
#> roc / auc


##### ==========================================================================
##### > Ré-entraîner Ranger
##### ==========================================================================
control <- trainControl(method="cv",number=4)

best_params <- data.frame()
j=1
for (j in 1:nrow(gr.foret)) {
    cat('Foret - ', j, '\n')
    tuned_model <- train(Y~.,data=donApp,
                         method="ranger",
                         classification = T,
                         metric = 'Accuracy',
                         trControl=control,
                         tuneGrid=expand.grid(
                             mtry=c(1,3,5),
                             splitrule='gini',
                             min.node.size=c(1,3,5,8)),
                         num.trees=gr.foret$num.trees[j],
                         max.depth=gr.foret$max.depth[j])
    
    results = tuned_model$results
    
    params = results[which.max(results$Accuracy),] %>% 
        mutate(num.trees=gr.foret$num.trees[j],
               max.depth=gr.foret$max.depth[j])
    
    best_params = rbind(best_params, params)
}

param_optimaux = best_params[which.max(best_params$Accuracy),]

foret.finale=ranger(factor(Y)~., data=donApp,
                    classification=T, probability = T,
                    
                    num.trees=param_optimaux$num.trees,
                    max.depth = param_optimaux$max.depth,
                    min.node.size = param_optimaux$min.node.size,
                    mtry = param_optimaux$mtry
                    )

SCORE_final = data.frame('Y'=donTest.Y)

SCORE_final[,'foret_finale'] = predict(foret.finale, donTest)$predictions[,'1']

rocCV = roc(factor(Y)~., data=SCORE_final, quiet=T)

auc(rocCV)


#### ===========================================================================
#### > Ré-entraîner XGBoost
#### ===========================================================================

### 0. Sélection de tout le jeu d'apprentissage

indY = which(names(donApp)=='Y')
X_train_final = as.matrix(donApp[,-indY])
Y_train_final = as.numeric(donApp[[indY]])-1

xgb_data_final_train = xgb.DMatrix(data=X_train_final, label = Y_train_final)
getinfo(xgb_data_final_train,'label') # pour vérifier la composition du label (en 0/1)

### 1. Grille d'hyperparamètres

metric = 'logloss'  # ou 'logloss' --> penser à modifier dans la boucle d'HP

xgrid = expand.grid(
    max_depth = c(1,2,5),
    eta = c(0.1,0.05),
    lambda = c(0,1,5,10),
    alpha=c(0,0.5,1)
)


### 2. Boucle hyper-paramètres

for(pp in 1:nrow(xgrid)){
    
    cat('\n',100*pp/nrow(xgrid),'% \n')
    
    params = list(
        objective = "binary:logistic",
        eval_metric = metric,
        max_depth = xgrid$max_depth[pp],
        eta = xgrid$eta[pp]
    )
    
    nIter = 900
    
    tmp = xgb.cv(
        params = params,
        nrounds = nIter,
        nfold = 5,
        verbose = F,
        early_stopping_rounds = 10,
        data=xgb_data_train
    )
    
    bestIter = tmp$best_iteration
    bestLogloss = tmp$evaluation_log$test_logloss_mean[bestIter]

    if(bestIter==nIter){
      cat('\n',"ATTENTION: BORD DE GRILLE: ", bestIter, '\n')
    } else{
        cat('\n' ,'Meilleure itération XGBoost : ', bestIter, '\n' )
    }
    
    x = data.frame(cbind(xgrid[pp,],bestIter,bestLogloss))
    
    xgrid[pp,'bestIter'] = bestIter  
    xgrid[pp,'bestLogloss'] = bestLogloss

}

xgrid[which.min(xgrid$bestLogloss),!names(xgrid)%in% c('bestLogloss')]
final.params = xgrid[which.min(xgrid$bestLogloss),!names(xgrid)%in% c('bestLogloss')]

nrounds = final.params$bestIter

xgb_final = xgb.train(objective = 'binary:logistic',
                      data = xgb_data_final_train,
                      max_depth = final.params$max_depth,
                      lambda = final.params$lambda,
                      alpha = final.params$alpha,
                      eta = final.params$eta,
                      nrounds = nrounds)



# Test et performances sur le dataset de test final

X_test_final = as.matrix(donTest)
Y_test_final = as.numeric(donTest.Y)-1
xgb_data_final_test = xgb.DMatrix(data=X_test_final)

SCORE_final_xgb=data.frame('Y'=Y_test_final,'xgb'=NA)

SCORE_final_xgb[,'xgb']=predict(xgb_final, newdata=xgb_data_final_test, type='response')
SCORE_final_xgb

rocCV_xgb = roc(factor(Y)~., data=SCORE_final, quiet=T)

auc(rocCV_xgb)


#===============================================================================
# > Feature engineering
#=============================================================================== 

### 0. Bases 
donApp
donTest

### 1. Polynomes
carre = function(x){return(x^2)}
cube = function(x){return(x^3)}

index.num.app = sapply(donApp, is.numeric)
index.num.app
index.num.test= sapply(donTest,is.numeric)
index.num.test

donApp.squared = sapply(donApp[,index.num.app],carre)
colnames(donApp.squared) = paste0('sqd_',colnames(donApp.squared))
donApp.cubed   = sapply(donApp[,index.num.app],cube)
colnames(donApp.cubed) = paste0('cub_',colnames(donApp.cubed))

donTest.squared = sapply(donTest[,index.num.test], carre)
colnames(donTest.squared) = paste0('sqd_',colnames(donTest.squared))
donTest.cubed = sapply(donTest[,index.num.test],cube)
colnames(donTest.cubed) = paste0('cub_',colnames(donTest.cubed))

donApp.poly = data.frame(cbind(donApp,donApp.squared, donApp.cubed))
donTest.poly = data.frame(cbind(donTest,donTest.squared, donTest.cubed)) 
donTest.poly$Y = donTest.Y

# Extraction des labels
donApp.Y = as.numeric(donApp$Y) - 1
donTest.Y

# Forme matricielle App
mat.donApp.poly = model.matrix(Y~., data= donApp.poly) 
train_features = colnames(mat.donApp.poly)


# Forme matricielle Test
mat.donTest.poly = model.matrix(Y~., data=donTest.poly)


xgb.app.poly = xgb.DMatrix(data=mat.donApp.poly, label=donApp.Y)
xgb.test.poly = xgb.DMatrix(data=mat.donTest.poly, label=donTest.Y)

SCORE.poly = data.frame('Y' = donTest.Y)

nIter = 50

xgb_final = xgb.train(objective = 'binary:logistic',
                      eval_metric = 'logloss',
                      data = xgb.app.poly,
                      max_depth = final.params$max_depth,
                      lambda = final.params$lambda,
                      alpha = final.params$alpha,
                      eta = final.params$eta,
                      nrounds = nrounds)

SCORE.poly[,'predict_poly'] = predict(xgb_final, newdata = xgb.test.poly)

rocCV = roc(factor(Y)~., data=SCORE.poly, quiet=T)

auc(rocCV)







