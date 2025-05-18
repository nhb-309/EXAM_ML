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

RES <- data.frame(Y_obs = Y_FIN,Y_pred= Y_pred_fin)

erreur2 = function(X,Y){mean((X-Y)^2)}

(apply(X = RES,MARGIN = 2,FUN = erreur2,Y=RES[,'Y_obs']))[-1]

# RF a gagné ----

## Création des matrices train et tests finales

foret <- randomForest(Y~.,data=don)
RES[,"foret"] <- predict(foret,donnees_test_final)

Y_pred_fin <- predict(xgbopt,xgb_data_test_final)

RES <- data.frame(Y_obs = Y_FINAL,Y_pred= Y_pred_fin)

erreur2 = function(X,Y){mean((X-Y)^2)}

(apply(X = RES,MARGIN = 2,FUN = erreur2,Y=RES[,'Y_obs']))[-1]

sort((apply(RES,2,erreur2,Y=RES[,'Y'])[-1]))

# RF Hyperparam a gagné ----

# nouveau train test pour hyperparametrage sur toutes les donnees train 
dataset_split <- initial_split(don,prop = 0.9,strata = Y)

donA <- training(dataset_split)

donT <- testing(dataset_split)


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
                          data=don,
                          ntree=best.foret.params$ntree,
                          mtry = best.foret.params$mtry,
                          nodesize= best.foret.params$nodesize)


SCORE[bloc==ii, 'best foret']= predict(best.foret, donnees_test_final)

