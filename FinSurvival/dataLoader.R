library(readr)
library(stringr)
library(dplyr)


## Helper functions
not_all_na <- function(x) any(!is.na(x))
`%notin%` <- Negate(`%in%`)


loadSurvivalDataset <- function(indexEvent, outcomeEvent, 
                                dataPath = "/data/IDEA_DeFi_Research/Data/Survival_Data/", 
                                X_path = "/X_train/",
                                y_path = "y_train.rds"){
  
  
  X_files <- list.files(paste0(dataPath, str_to_title(indexEvent), X_path), pattern = NULL, all.files = FALSE, full.names = TRUE)
  X = data.frame()
  
  for(file in X_files){
    if(grepl("pca_result.rds", file)){
      next
    }
    X <- X %>%
      bind_rows(read_rds(file)) %>%
      select(where(not_all_na)) %>%
      select(-starts_with("exo")) %>%
      filter(!is.na(id))
  }
  
  y <- read_rds(paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/", y_path)) %>%
    filter(!is.na(id))
  
  
 # for flattened transactions: 
 # return(inner_join(y, X, by = c("id" = "id_transaction_1")))
 # for LTM
  return(inner_join(y, X, by = c("id")))
}

train = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_train_flattened/", y_path = "y_train.rds")
test = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_test_flattened/", y_path = "y_test.rds")

#train = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_raw_seqLen10/", y_path = "y_train.rds")
#test = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_raw_seqLen10/", y_path = "y_test.rds")


#train = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_ltm_small/", y_path = "y_train.rds")
#test = loadSurvivalDataset(indexEvent, outcomeEvent, X_path = "/X_ltm_small/", y_path = "y_test.rds")


# Find the shared columns
shared_columns <- intersect(names(train), names(test))

# Subset each dataframe
train <- train %>% select(all_of(shared_columns))
test <- test %>% select(all_of(shared_columns))
