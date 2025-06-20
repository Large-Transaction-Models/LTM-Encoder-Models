library(fastDummies)
library(dplyr)
library(conflicted)
library(caret)

conflict_prefer("filter", "dplyr")
conflict_prefer("summarize", "dplyr")
conflict_prefer("select", "dplyr")

limitFactorLevels <- function(df, maxLevels = 10, padLabel = "PAD") {
  #---------------------------------------------------------------------------
  # PURPOSE  :  (i)   cap the number of levels of every categorical column at
  #                    `maxLevels`, lumping the remainder into "Other";
  #             (ii)  convert *structural* missing values (NA) to an explicit
  #                    level  'padLabel' (default "PAD").
  #
  # RETURNS  :  list(
  #               transformedData = data.frame with modified factors,
  #               factorLevels    = named list of the final levels per column
  #             )
  #
  # NOTE     :  By turning NA into a real level we keep “no prior transaction”
  #             information available to downstream models and in parity with
  #             your LTM’s PAD token.
  #---------------------------------------------------------------------------
  
  # container for the retained levels of every factor
  factorLevelInfo <- list()
  
  # copy to avoid mutating the caller’s data frame by reference
  dfModified <- df
  
  # automatically detect categorical columns (factor OR character)
  categoricalCols <- names(df)[sapply(df, function(col) is.factor(col) || is.character(col))]
  
  for (colName in categoricalCols) {
    
    # ---------- 1. Replace NA with the explicit PAD label ----------
    colData <- as.character(dfModified[[colName]])    # coerce to character so we can edit
    colData[is.na(colData)] <- padLabel
    dfModified[[colName]] <- colData                  # write back (still character)
    
    # ---------- 2. Compute frequency counts & keep `maxLevels` modes ----------
    freqTable <- sort(table(dfModified[[colName]]), decreasing = TRUE)
    topCategories <- names(freqTable)[seq_len(min(maxLevels, length(freqTable)))]
    
    # ---------- 3. Re‑encode column, lumping rare levels into "Other" ----------
    # Convert again to character to allow assignment
    dfModified[[colName]] <- as.character(dfModified[[colName]])
    
    isTop <- dfModified[[colName]] %in% topCategories
    isPad <- dfModified[[colName]] == padLabel        # padLabel is *never* lumped
    
    # Replace only those that are neither top nor PAD with "Other"
    dfModified[[colName]][!(isTop | isPad)] <- "Other"
    
    # ---------- 4. Final factor conversion with a stable level order ----------
    finalLevels <- unique(c(topCategories, "Other", padLabel))  # preserve order
    dfModified[[colName]] <- factor(dfModified[[colName]], levels = finalLevels)
    
    # store the mapping for future test‑set transformation
    factorLevelInfo[[colName]] <- finalLevels
  }
  
  list(
    transformedData = dfModified,
    factorLevels    = factorLevelInfo
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
    encodedList[['factors']] <- trsf
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
                   "user", "timestamp", "...1")
  
  train <- train %>%
    select(-any_of(cols_to_drop),
           -starts_with("id"),
           -starts_with("pool")) %>%
    mutate(across(where(is.character), as.factor))
  test <- test %>%
    select(-any_of(cols_to_drop),
           -starts_with("id"),
           -starts_with("pool")) %>%
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
  
  # Convert NA numeric features to 0:
  train[is.na(train)] <- 0
  test[is.na(test)] <- 0
  
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
    pca_result <- prcomp(train, center = FALSE, scale. = FALSE, rank. = 500)
    
    
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
  
  
  
  
  