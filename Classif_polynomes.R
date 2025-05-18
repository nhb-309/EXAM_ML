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

# Répertoire ----
setwd()

# Import ----
db = SAheart %>% 
    rename(Y=chd) %>% 
    mutate(Y=as.factor(Y)) %>% 
    mutate(across(.cols=where(is.character),.fns=as.factor)) %>% 
    mutate(famhist = case_when(famhist=='Present'~'1',
                               famhist=='Absent'~'0')) 

# Notes ----
#> il faut faire attention à ce que toutes les variables categorielles soient 
#> passées en facteur 
#>

# lecture
db = read.csv('bank.csv',sep=';')  %>% 
    rename(Y=y) %>%
    mutate(Y = case_when(Y=='no'~0,
                         Y=='yes'~1)) %>% 
    mutate(across(.cols = where(is.character),.fns=as.factor)) %>% 
    mutate(Y=as.factor(Y))

#> Vérifier que Y n'est pas considéré comme numérique! 
#> il ne faut surtout pas passer Y. 
var.num = sapply(db, is.numeric)
#> contrôler!
db[,var.num] %>% head()

# Retrait des NA, mais uniquement s'il y en a peu
db = db %>% na.exclude()
# strings as factors
db = db %>% mutate(across(.cols=where(is.character),.fns=as.factor))

# (Summary) ----
summary(db)
sapply(db, class)
boxplot(db)
db %>% str()

# Graine
set.seed(12)

# Split ----

dbsplit = initial_split(db, prop=0.9, strata=Y)
donApp = training(dbsplit)
donTest = testing(dbsplit)

donTest.Y = donTest$Y

donTest = donTest %>% select(-Y)

# Degrés 2 et 3 ----
var.num = sapply(donApp, is.numeric)
carre = function(x){return(x^2)}
cube = function(x){return(x^3)}

donApp.carre = sapply(donApp[,var.num], carre)
colnames(donApp.carre) = paste0('car_',colnames(donApp[,var.num]))
donApp.cub=sapply(donApp[,var.num], cube)
colnames(donApp.cub) = paste0('cub_',colnames(donApp[,var.num]))

donApp.poly = cbind(donApp,donApp.carre,donApp.cub)

donTest.carre = sapply(donTest[,var.num], carre)
colnames(donTest.carre) = paste0('car_',colnames(donTest[,var.num]))
donTest.cub = sapply(donTest[,var.num], cube)
colnames(donTest.cub) = paste0('cub_', colnames(donTest[,var.num]))

donTest.poly = cbind(donTest,donTest.carre,donTest.cub)

# Recettes avec tidymodels ----
recipe.APP = recipe(Y ~ ., data = donApp.poly) %>% 
    step_zv(all_predictors()) %>% 
    step_normalize(all_numeric_predictors()) %>% 
    step_impute_median(all_numeric_predictors()) %>% 
    step_impute_mode(all_nominal_predictors()) %>% 
    step_dummy(all_nominal_predictors(),one_hot = T) 

recipe.TEST = recipe(~ ., data = donTest) %>% 
    step_zv(all_predictors()) %>% 
    step_normalize(all_numeric_predictors()) %>% 
    step_impute_median(all_numeric_predictors()) %>% 
    step_impute_mode(all_nominal_predictors()) %>% 
    step_dummy(all_nominal_predictors(),one_hot=T) 

# Prep / Bake ----
poly.app = recipe.APP %>% prep() %>% bake(new_data=donApp.poly)
poly.test= recipe.TEST %>% prep() %>% bake(new_data=donTest.poly)

# Matrices de modèle ----
mat.poly.app.X = model.matrix(Y~., data=poly.app)[,-1]
mat.poly.app.Y = donApp$Y

nbloc=3

folds = createFolds(donApp$Y,
                    k=nbloc,
                    list=F) # a voir si createFolds garde la proportion automatiquement

# Grilles ---- 
## Grille Foret ----
gr.foret=expand.grid(num.trees=c(100,300),
                     max.depth=c(1,5))

gr.foret.params=data.frame(gr.foret,'auc'=NA)

## Grille SVM ----
gr.poly=expand.grid(C=c(0.1,1),
                    degree=c(1,2,3),
                    scale=1)
gr.radial=expand.grid(C=c(0.1,1),
                      sigma = c(0.0001,0.1,1))

ctrl = trainControl(method='cv', number=5)

## Grille XGBoost ----
xgrid = expand.grid(
    max_depth = c(1,5),
    eta = c(0.1),
    reg_lambda = c(0,10),
    reg_alpha=c(1)
)

# ____________________________________ ----

# Validation croisee ----
SCORE=data.frame('Y'=donApp$Y,'logistic'=NA,'aic'=NA,'bic'=NA,'ridge'=NA,'lasso'=NA,'elnet'=NA,'foret'=NA,'rad_svm'=NA,'pol_svm'=NA,'gbm'=NA,'xgb'=NA)

jj=1
for(jj in 1:nbloc){
    cat('Fold: ', jj, '\n')
    
    donA=poly.app[folds!=jj,]
    donV=poly.app[folds==jj,]
    
    donXA=mat.poly.app.X[folds!=jj,]
    donXV=mat.poly.app.X[folds==jj,]  
    donYA=donApp$Y[folds!=jj]
    
    
    ## Logistique ----
    logistic=glm(Y~., data=donA, family='binomial')
    SCORE[folds==jj,'logistic'] = predict(logistic,newdata=donV,type='response')
    
    ## AIC ----
    #aic=stats::step(logistic,trace=0)
    #SCORE[folds==jj,'aic']=predict(aic,newdata=donV,type='response')
    
    ## BIC ----
    #bic=stats::step(logistic,trace=0,k=log(nrow(donA)))
    #SCORE[folds==jj,'bic']=predict(bic,newdata=donV,type='response')
    
    ## Penalisation ----
    
    ridge=cv.glmnet(donXA,donYA,alpha=0  ,family='binomial',nfolds=5,type.measure='auc')
    lasso=cv.glmnet(donXA,donYA,alpha=1  ,family='binomial',nfolds=5,type.measure='auc')
    elnet=cv.glmnet(donXA,donYA,alpha=0.5,family='binomial',nfolds=5,type.measure='auc')
    SCORE[folds==jj,'ridge']=predict(ridge,newx=donXV,type='response',s='lambda.min')
    SCORE[folds==jj,'lasso']=predict(lasso,newx=donXV,type='response',s='lambda.min')
    SCORE[folds==jj,'elnet']=predict(elnet,newx=donXV,type='response',s='lambda.min')
    
    ## Foret ----
    
    ### Optimisation ----
    
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
    
    ### Hyper-paramètres optimaux ----
    param_optimaux = best_params[which.max(best_params$Accuracy),]
    
    foret.finale=ranger(factor(Y)~., data=donA,
                        classification=T, probability = T,
                        num.trees=param_optimaux$num.trees,
                        max.depth = param_optimaux$max.depth,
                        min.node.size = param_optimaux$min.buckets,
                        mtry = param_optimaux$mtry)
    
    SCORE[folds==jj,'foret'] = predict(foret.finale,data=donV,type='response')$predictions[,'1']
    
    
    
    ## SVM ----
    ### SVM Poly ----
    svm.poly = train(Y~., data=donA, method = 'svmPoly', trControl = ctrl, tuneGrid = gr.poly)
    bestPoly=svm.poly$results[which.max(svm.poly$results$Accuracy),]
    tmpPol=ksvm(Y~., data=donA, kernel = 'polydot', kpar=list(degree=bestPoly$degree,scale=1,offset=1),C=bestPoly$C,prob.model=T)
    SCORE[folds==jj,'pol_svm'] = predict(tmpPol, newdata=donV, type = 'prob')[,'1'] # rajouter l'index de colonne : il ne faut en prendre qu'une
    
    ### SVM Radial ----
    svm.radial = train(Y~., data=donA, method = 'svmRadial', trControl = ctrl, tuneGrid = gr.radial)
    bestRadial=svm.radial$results[which.max(svm.radial$results$Accuracy),]
    tmpRad=ksvm(Y~., data=donA, kernel =  'rbfdot', kpar=list(sigma=bestRadial$sigma),C=bestRadial$C,prob.model=T)
    SCORE[folds==jj,'rad_svm'] = predict(tmpRad, newdata=donV, type= 'prob')[,'1']  
    
    ## Gradient Boosting ----
    donA.gbm = donA %>% mutate(Y = as.numeric(Y) - 1)
    tmp <- gbm(Y~.,data=donA.gbm,distribution = "bernoulli",cv.folds=5,n.trees=300,
               interaction.depth = 3,shrinkage = 0.1)
    best.iter <- gbm.perf(tmp, method = "cv")
    SCORE[folds==jj,"gbm"] = predict.gbm(tmp,donV,type='response',n.trees = best.iter)
    
    ## XGBoost ---- 
    indY = which(names(donA)=='Y')
    
    X_train = as.matrix(donA[,-indY])
    Y_train = as.numeric(donA[[indY]])-1
    
    X_test = as.matrix(donV[,-indY])
    Y_test = as.numeric(donV[[indY]])-1
    
    xgb_data_train=xgb.DMatrix(data = X_train, label = Y_train)
    xgb_data_test=xgb.DMatrix(data = X_test)
    
    results = data.frame()
    
    for(pp in 1:nrow(xgrid)){
        
        cat('\n',100*pp/nrow(xgrid),'% \n')
        
        params = list(
            objective = "binary:logistic",
            metrics = "logloss",
            stratified = T,
            max_depth = xgrid$max_depth[pp],
            alpha = xgrid$reg_alpha[pp],
            lambda = xgrid$reg_lambda[pp],
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
        
        bestLogloss = tmp$evaluation_log$test_logloss_mean[bestIter]
        
       # x = data.frame(cbind(xgrid[pp,],bestIter,bestLogloss))
        results[pp,1:ncol(xgrid)] = xgrid[pp,]
        results[pp,'bestIter'] = bestIter
        results[pp,'bestLogloss'] = bestLogloss
    }
    
    best_params_index <- results[which.min(results$bestLogloss),] 
    
    checkIter = best_params_index$bestIter
    
    #if(checkIter == nIter){
    #    cat('\n', "ATTENTION: BORD DE GRILLE --> ", checkIter,'\n')
    #}else{
    #    cat('\n','Meilleure itération XGBoost: ', checkIter,'\n')
    #}
    
    best_params = list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = best_params_index$max_depth,
        eta = best_params_index$eta,
        alpha = best_params_index$reg_alpha,
        lambda = best_params_index$reg_lambda
    )
    
    xgb = xgb.train(params = best_params,
                    data = xgb_data_train, 
                    nrounds = best_params_index$bestIter)
    
    SCORE[folds==jj, 'xgb']=  predict(xgb,newdata = xgb_data_test,type='response')
    
}
# _____________________________________ ----

# Analyse des scores ----
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






