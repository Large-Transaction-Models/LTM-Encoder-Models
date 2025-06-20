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
  
  
  X <- read_rds(paste0(dataPath, str_to_title(indexEvent), X_path)) %>%
    filter(!is.na(id))
  
  y <- read_rds(paste0(dataPath, str_to_title(indexEvent), "/", str_to_title(outcomeEvent), "/", y_path)) %>%
    filter(!is.na(id)) %>%
    mutate(id = as.numeric(id))
  
  
  
  return(inner_join(y, X, by = "id"))
  
}
