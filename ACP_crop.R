library(FactoMineR)


#################################################################################
#################################################################################
# en pratique: 


#> Analyse des variables: chercher à comprendre ce qu'on trouve sur le cercle des 
#> corrélations entre variables. 

#> 0. Données

don = read.csv('crop.csv', sep=',')[,!names(don) %in% c('label')]

#> 1. Calcul de l'ACP et affichage du cercle

res.pca = PCA(don)

#> les deux dimensions capturent une part assez faible de la variance globale des variables. 
barplot(res.pca$eig[,2], names=paste("Dim", 1:nrow(res.pca$eig))) # 4 dimensions ça serait mieux. 

#> mais si on regarde les flèches sur le graphique des variables on voit que: 
#> K et P (ammoniaque et potassium) --> longues flèches assez proches ça veut dire que ces deux variables sont 
#> bien capturées par les deux dimensions et sont assez corrélées entre elles. Autrement dit 
#> ce sont des cultures en sol riches en nutriments. A l'opposé de cultures nécessitant un sol plus riche en pH et en azote.
#> mais ph et azote sont assez mal capturées par ces deux dimensions. 
#> humidité est également très bien capturée par les deux dimensions 

# Pour confirmer cette lecture graphique on va chercher les variables bien représentées 
# sur Dim 1 (cos2 > 0.5) puis sur Dim 2
which(res.pca$var$cos2[,1] > 0.5)
which(res.pca$var$cos2[,2] > 0.5) # s'il n'y en a pas --> chercher le top trois des mieux représentées
sort(res.pca$var$cos2[,2], decreasing=T)[1:3]

# Interprétation: les variables les mieux représentées à l'aide du cos2 sont celles
# qui permettent de décrire le mieux la dimension

###
# 2.2. Contribution de la variable sur les dimensions 1 et 2.
###

# ici Potassium contribue à hauteur de 41% de la dim 1.  
# en gros la dim 1 capture 27,6% de la variance totale des données. 
# 41% * 27% = 11% --> P explique 11% de la variance totale du dataset. 
sort(res.pca$var$contrib[,1], decreasing = TRUE)[1:3] 
sort(res.pca$var$contrib[,2], decreasing = TRUE)[1:3] 

# Interprétation : ce sont les variables qui contribuent le plus à la construction
# de la dimension. 


# 3. Affichage des individus projetés sur la dim 1 et dim 2
# Permet de situer chaque individu sur les dimensions et donc en déduire leurs qualités
# ou performances associées. 
plot(res.pca, choix = 'ind', 
     select="cos2 0.2", title = 'Graphe des individus',label = 'none')
# !!!! argument important: le select cos2 0.6.
#> explication: on rappelle que l'acp permet la synthèse d'un grand nombre de dimensions en deux dimensions cad sur un plan.
#> plus le point d'un individu est proche du plan de l'acp plus les deux dimensions de ce plan capturent une part importante 
#> de la variance totale de cet individu dans l'espace à p dimensions du jeu de données.  
#> 
#> cos2 élevé => la projection de l'individu sur le plan de l'ACP est fidèle à sa position relative dans l'espace à d dims. 
#> cos2 faible => la projection de l'individu sur le plan de l'ACP n'est pas fiable, 



