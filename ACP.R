library(FactoMineR)

#> L'objectif d'une ACP est de résumer et visualiser un tableau de données individus x variables. 
#> L'ACP permet d'étudier les ressemblances entre individus du point de vue de 
#> l'ensemble des variables et dégage des profils d'individus. Elle permet également 
#> de réaliser un bilan des liaisons linéaires entre variables à partir des coefficients 
#> de corrélations. Ces études étant réalisées dans un même cadre, elles sont liées, 
#> ce qui permet de caractériser les individus ou groupes d'individus par les variables et 
#> d'illustrer les liaisons entre variables à partir d'individus caractéristiques. 
#> 
#> Soit on fait : 
#> - une étude des variables : quelles ont les variables qui apportent une information identique/différente ?
#>      Avec le coefficient de corrélation on peut trouver des variables qui apportent la même information (corr~1)
#>      Ou 
#> - une étude des individus : comment peut on regrouper les individus compte tenu des variables dont on dispose ?

decath = read.table('decathlon.csv', sep=';', dec = '.', header = T, row.names=1, check.names=F)

#> Question : déterminer des profils de performances: il faut mettre en actif les 
#> variables correspondant aux performances des athlètes. Le choix des variables 
#> actives est très important: ce sont ces variables et uniquement ces variables
#> qui participent à la construction des axes de l'ACP. 
#> 
#> On peut aussi ajouter en variables supplémentaires les variables quantitatives 
#> - nombre de points
#> - classement
#> et qualitative: - compétition
#> Les variables supplémentaires sont très utiles pour aider à interpréter les axes.
#> On choisit aussi les individus actifs : ceux qui participent à la construction des axes. 
#> Ici comme fréquemment, tous les individus sont considérés comme actifs. 

#> Standardiser ou non les variables: centrer-réduire ou juste centrer ?
#> Si unités différentes => c'est indispensable de centrer-réduire. 
#> La réduction permet d'accorder la même importance à chacune des variables 
#> alors que ne pas réduire revient à donner à chaque variable un poids 
#> correspondant à son écart-type. Ce choix est d'autant plus important 
#> que les variables ont des variances très différentes. 

res.pca = PCA(decath, quanti.sup = 11:12, quali.sup = 13)

#> par défaut: la fonction PCA de FactoMineR opère un centrage-réduction des données.
#> Si on veut le désactiver : scale.unit = F

sapply(decath, stats::var)

#> Choisir le nombre d'axes: rechercher une rupture
barplot(res.pca$eig[,2], names=paste("Dim", 1:nrow(res.pca$eig)))
#> une barre par dimension : 
summary.PCA(res.pca)
#> Les deux premiers axes expriment 50% de l'inertie totale c'est à dire que 50% de 
#> l'information du tableau de données est contenue dans les deux premières dimensions. 
#> Cela signifie que la diversité des profils de performances ne peut être
#> résumée par deux dimensions. Ici, les quatre premiers axes permettent d'expliquer 
#> 75% de l'inertie totale. 
#> 

res.pca = PCA(decath, quanti.sup = 11:12, quali.sup = 13)
plot(res.pca, choix = 'ind', habillage = 13, cex = 1.1, 
     select="cos2 0.6", title = 'Graphe des individus')



#################################################################################
#################################################################################
# en pratique: 


#> Analyse des variables: chercher à comprendre ce qu'on trouve sur le cercle des 
#> corrélations entre variables. 

#> 0. Données 

decath

#> 1. Calcul de l'ACP et affichage du cercle

res.pca = PCA(decath, quanti.sup = 11:12, quali.sup = 13)

#> Flèches proches les unes des autres: variables corrélées
#> Flèches vers la gauche proche DIM 1 = variables fortement négative sur Dim1 (faire gaffe à l'unité de chaque var)
#> # ex: la dimensions 1 rassemble des variables liées à un effort explosif. 
#> # car pour le 100m (flèche proche de DIM 1 et vers la gauche) on a des temps élevés (moins explosifs) 
#> # autrement dit des temps élevés au 100m = faibles performances de sprint. 
#> # idem pour la longueur : c'est une variable de distance vers la droite 

# et on regarde combien de dimensions contiennent la majorité de l'information
barplot(res.pca$eig[,2], names=paste("Dim", 1:nrow(res.pca$eig)))

#> 2. Plus quantitativement on regarde la contribution de chaque variable aux 
#> deux premières dimensions

###
# 2.1. Représentation de la variable sur les dimensions 1 et 2.
###

# Variables bien représentées sur Dim 1 (cos2 > 0.5) puis sur Dim 2
which(res.pca$var$cos2[,1] > 0.5)
which(res.pca$var$cos2[,2] > 0.5) # s'il n'y en a pas --> chercher le top trois des mieux représentées
sort(res.pca$var$cos2[,2], decreasing=T)[1:3]

# Interprétation: les variables les mieux représentées à l'aide du cos2 sont celles
# qui permettent de décrire le mieux la dimension

###
# 2.2. Contribution de la variable sur les dimensions 1 et 2.
###

sort(res.pca$var$contrib[,1], decreasing = TRUE)[1:3]
sort(res.pca$var$contrib[,2], decreasing = TRUE)[1:3]

# Interprétation : ce sont les variables qui contribuent le plus à la construction
# de la dimension. 


# 3. Affichage des individus projetés sur la dim 1 et dim 2
plot(res.pca, choix = 'ind', habillage = 13, cex = 1.1, 
     select="cos2 0.6", title = 'Graphe des individus')

# Permet de situer chaque individu sur les dimensions et donc en déduire leurs qualités
# ou performances associées. 

# Bourguignon et Uldal font des temps élevés aux épreuves de 100m ou 110m... donc pas fous dans ces disciplines. 
# Alors que Sebrle, Clay et Karpov font des temps très courts sur ces mêmes disciplines. 
# Casarsa est éclaté en 1500m alors que Drews fait des temps assez faibles sur cette épreuve. 
# Drews est bon en endurance et moyen sur la Dim 1 soit les épreuves un peu explosives type 100m et saut en longueur. 




