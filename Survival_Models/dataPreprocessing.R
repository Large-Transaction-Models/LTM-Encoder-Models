library(fastDummies)
library(dplyr)
library(conflicted)

conflict_prefer("filter", "dplyr")
conflict_prefer("summarize", "dplyr")
conflict_prefer("select", "dplyr")

limitFactorLevels <- function(df, maxLevels = 10) {
  # List to store the top levels for each factor
  factorLevelInfo <- list()
  
  # Make a copy to avoid mutating original data
  dfModified <- df
  
  # Find categorical columns automatically (factors or characters)
  categoricalCols <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
  
  # Process each categorical column
  for(col in categoricalCols) {
    # Compute frequency of each category
    freqTable <- sort(table(df[[col]]), decreasing = TRUE)
    
    # Extract top categories (up to maxLevels)
    topCategories <- names(freqTable)[seq_len(min(length(freqTable), maxLevels))]
    
    if(length(topCategories) == maxLevels){
      # Assign "Other" to categories not in topCategories
      dfModified[[col]] <- as.character(dfModified[[col]])
      dfModified[[col]][!(dfModified[[col]] %in% topCategories)] <- "Other"
      
      # Convert to factor with consistent ordering (top categories first, then "Other")
      finalLevels <- c(topCategories, "Other")
      dfModified[[col]] <- factor(dfModified[[col]], levels = finalLevels)
      
    }
    # Save the factor levels for applying later to test data
    factorLevelInfo[[col]] <- finalLevels
  }
  
  # Return both the modified train set and the factor level mapping
  list(
    transformedData = dfModified,
    factorLevels = factorLevelInfo
  )
}

applyFactorLevels <- function(df, factorLevels) {
  dfModified <- df
  
  for(col in names(factorLevels)) {
    if(col %in% names(dfModified)) {
      # Set categories not in the factorLevels to "Other"
      dfModified[[col]] <- as.character(dfModified[[col]])
      dfModified[[col]][!(dfModified[[col]] %in% factorLevels[[col]])] <- "Other"
      
      # Convert to factor using stored levels
      dfModified[[col]] <- factor(dfModified[[col]], levels = factorLevels[[col]])
    }
  }
  
  dfModified
}

oneHotEncode <- function(df) {
  # Identify factor columns to encode
  factorCols <- names(df)[sapply(df, is.factor)]
  
  # Numeric columns to keep unchanged
  numericCols <- setdiff(names(df), factorCols)
  
  # Initialize a list to store encoded columns
  encodedList <- list()
  
  # One-hot encode factor columns
  if (length(factorCols) > 0) {
    dmy <- dummyVars(" ~ .", data = df)
    trsf <- data.frame(predict(dmy, newdata = df))
  }
  
  # Combine numeric columns if any exist
  if (length(numericCols) > 0) {
    encodedList[['numerics']] <- df[numericCols]
  }
  
  # Combine all encoded columns into a single dataframe
  dfEncoded <- do.call(cbind, encodedList)
  
  return(dfEncoded)
}


preprocess <- function(train, test,
                       useScaling = TRUE,
                       useOneHotEncoding = TRUE,
                       usePCA = TRUE, pcaExplainedVar = 0.9, 
                       classificationTask = FALSE, classificationCutoff = -1){
  
  
  # Let's save off the target columns up front so we can drop them before scaling:
  trainTargets <- train %>%
    select(timeDiff, status)
  testTargets <- test %>%
    select(timeDiff, status)
  # Let's drop some of the columns that we know we can't use:
  cols_to_drop = c("timeDiff", "status",
                   "id", "Index Event", "Outcome Event",
                   "type", "pool",
                   "user", "timestamp")
  
  train <- train %>%
    select(-any_of(cols_to_drop)) %>%
    mutate(across(where(is.character), as.factor))
  test <- test %>%
    select(-any_of(cols_to_drop)) %>%
    mutate(across(where(is.character), as.factor))
  
  
  
  trainDataCategoricalCols <- train %>%
    select(where(is.factor))
  testDataCategoricalCols <- test %>%
    select(where(is.factor))
  train <- train %>%
    select(-where(is.factor))
  test <- test %>%
    select(-where(is.factor))
  
  if(useScaling == TRUE){
    # Let's scale the data:
    train <- scale(as.matrix(train))
    
    test <- data.frame(scale(as.matrix(test), center=attr(train, "scaled:center"), scal=attr(train, "scaled:scale")))
    
    train <- data.frame(train) 
    test <- data.frame(test)
    
    train <- train[ , !sapply(train, function(x) all(is.na(x)))]
    
    common_cols <- intersect(colnames(train), colnames(train))
    test <- test %>%
      select(all_of(common_cols))
    
  }
  
  if(useOneHotEncoding == TRUE){
    outputs <- limitFactorLevels(trainDataCategoricalCols)
    trainDataCategoricalCols = outputs[[1]]
    factorLevels <- outputs[[2]]
    testDataCategoricalCols = applyFactorLevels(testDataCategoricalCols, factorLevels)
    trainDataCategoricalCols <- oneHotEncode(trainDataCategoricalCols)
    testDataCategoricalCols <- oneHotEncode(testDataCategoricalCols)
  }
  
  # Put categorical columns back in the data:
  train <- train %>%
    bind_cols(trainDataCategoricalCols)
  
  test <- test %>%
    bind_cols(testDataCategoricalCols)
  
  
  
  
  if(usePCA == TRUE){
    # Now that we have scaled, encoded data, let's run PCA to help eliminate collinearity of features:
    pca_result <- prcomp(train, center = FALSE, scale. = FALSE)
    
    
    # Variances of each PC are the squared standard deviations:
    pc_variances <- pca_result$sdev^2  
    
    # Proportion of total variance explained by each PC:
    prop_variance <- pc_variances / sum(pc_variances)
    
    # Cumulative variance explained:
    cumvar <- cumsum(prop_variance)
    
    # Find the smallest number of PCs explaining at least 90% variance:
    num_pcs <- which(cumvar >= pcaExplainedVar)[1]
    
    # Keep only the PCs that explain ≥ 90% of variance
    train <- as.data.frame(pca_result$x[, 1:num_pcs])
    
    # 'scores_90pct' is now a matrix with the same number of rows as df,
    # but fewer columns (one column per principal component).
    test <- data.frame(predict(pca_result, newdata = test))[, 1:num_pcs]
  }
  
  
  # Put the targets back in the data:
  final_train_data <- train %>%
    bind_cols(trainTargets)
  
  final_test_data <- test %>%
    bind_cols(testTargets)
  
 
  
  if(classificationTask){
    # filter out invalid records where `timeDiff` is <= 0 early
    final_train_data <- final_train_data %>% filter(timeDiff > 0)
    
    # filter out records based on the `set_timeDiff` threshold and `status`
    final_train_data <- final_train_data %>% filter(!(timeDiff / 86400 <= classificationCutoff & status == 0))
    
    # create a new binary column `event` based on `timeDiff`
    final_train_data <- final_train_data %>%
      mutate(event = case_when(
        timeDiff / 86400 <= classificationCutoff ~ "yes",
        timeDiff / 86400 > classificationCutoff ~ "no"
      )) %>%
      select(-timeDiff, -status)
    
    # filter out invalid records where `timeDiff` is <= 0 early
    final_test_data <- final_test_data %>% filter(timeDiff > 0)
    
    # filter out records based on the `set_timeDiff` threshold and `status`
    final_test_data <- final_test_data %>% filter(!(timeDiff / 86400 <= classificationCutoff & status == 0))
    
    # create a new binary column `event` based on `timeDiff`
    final_test_data <- final_test_data %>%
      mutate(event = case_when(
        timeDiff / 86400 <= classificationCutoff ~ "yes",
        timeDiff / 86400 > classificationCutoff ~ "no"
      )) %>%
      select(-timeDiff, -status)
  }
  
  return(list(final_train_data, final_test_data))
}
  
  
  
  
  