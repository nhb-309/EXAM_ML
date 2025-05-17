db = read.csv('crop.csv',sep=',')%>% select(-label)

my_recipe = recipe(~. , data=db) %>% 
    step_impute_median(all_numeric_predictors())%>% 
    step_normalize(all_numeric_predictors())


don = my_recipe %>% 
    prep() %>% 
    bake(new_data=db)

# Calcul de la matrice de distances 
mat_dist = dist(don,method='euclidean')

# Calcul de la CAH
cah=hclust(mat_dist, method= 'ward.D')

plot(rev(cah$height)[1:10], type='b')

don$cluster = cutree(cah, k=4)


# Calcul des centroïdes: 
centroids <- don %>%
    group_by(cluster) %>%
    summarise(across(everything(), mean)) %>%
    column_to_rownames("cluster")

# 2. Prepare and normalize new data
don_test <- bake(prep(my_recipe), new_data = db_test[, num_var])

# 3. Compute distances to each centroid and assign the nearest
assign_cluster <- function(row) {
    dists <- apply(centroids, 1, function(center) sum((row - center)^2))
    return(as.integer(names(which.min(dists))))
}

new_clusters <- apply(don_test, 1, assign_cluster)

don_test$cluster = new_clusters