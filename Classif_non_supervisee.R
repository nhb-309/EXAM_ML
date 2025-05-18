
# CLASSIFICATION NON-SUPERVISEE

data= crop %>% select(!label)

# Division en train et test ----

dataset_split <- initial_split(data,
                               prop = 0.9)

don <- training(dataset_split)

donnees_test_final <- testing(dataset_split)


# Méthode des kmeans ----

## Jeu d'apprentissage ----

### Normaliser les données ----
don_normalized <- scale(don)

### Trouver le nombre optimal de clusters (méthode du coude) ----

# calcul de l'inertie intra class pour chaque k
intra_class <- sapply(1:10, function(k) {
  kmeans(don_normalized, centers = k, nstart = 10)$tot.withinss
})

# Tracer le graphique du coude
data.frame(k = 1:10, ic = intra_class) %>%
  ggplot(aes(k, ic)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(n.breaks = 10)+
  ggtitle("Méthode du coude pour déterminer le nombre optimal de clusters")

# Trouver le nombre optimal de classes

# optimal_k <- which.min(diff(intra_class, lag = 1)) + 1
optimal_k=3

### Calcul des kmeans avec le k optimal ----
kmeans_result <- kmeans(don_normalized, centers = optimal_k)

# Afficher les résultats
print(kmeans_result)

# Visualisation Graphique
plot(don,col=kmeans_result$cluster)

## Jeu de Test ----

### Normaliser les nouvelles données ----
donnees_test_final_normalized <- scale(donnees_test_final)

### Assigner les nouvelles données aux clusters calculés ----

# Utiliser le modèle k-means pour prédire les clusters des nouvelles observations
predict.kmeans <- function(object, newdata) {
  centers <- object$centers
  apply(newdata, 1, function(row) {
    which.min(colSums((t(centers) - row)^2))
  })
}

new_clusters <- predict.kmeans(kmeans_result, donnees_test_final_normalized)

# Ajouter les clusters prédits aux nouvelles données
donnees_test_final_with_clusters <- donnees_test_final %>%
  mutate(Cluster = as.factor(new_clusters))

# Afficher les nouvelles données avec les clusters assignés
print(donnees_test_final_with_clusters)

# Visualisation Graphique
plot(donnees_test_final,col=kmeans_result$cluster)


# Méthode de la CAH ----

## Jeu d'apprentissage ----

### On crée d'abord une matrice de distance ----
mat <- dist(don_normalized, method='euclidian')

### Méthode d'aggrégation : ward ----
cah=stats::hclust(mat,method='ward.D2') # ward agrège en fonction de la distance au centre de gravité. 
plot(cah,labels=F)

### Identification du point de rupture ----
plot(rev(cah$height)[1:10], type='b')
plot(cah)

n.clusters = 5

don$cluster = cutree(cah, k=n.clusters)

don %>% head()

## Jeu de test ----


### répartir les données test en clusters ---- 

# Calcul des centroids
centroids <- don %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean)) %>%
  column_to_rownames("cluster")

centroids


# Calculer les distances aux centroides (euclidienne), et assigner au plus proche

assign_cluster <- function(row) {
  
  dists <- apply(centroids, 1, function(center) sum((row - center)^2))
  
  return(as.integer(names(which.min(dists))))
}

new_clusters <- apply(donnees_test_final_normalized, 1, assign_cluster)

donnees_test_final$cluster = new_clusters
