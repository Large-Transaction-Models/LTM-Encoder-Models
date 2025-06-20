aft_regression <- function(train, test,
                           features   = NULL,
                           dist       = "weibull",
                           maxiter    = 100,
                           rel.tol    = 1e-9,
                           nzv_cutoff = 0.95) {
  
  ## ------------------------------------------------------------------ ##
  ## 0.  Packages ------------------------------------------------------ ##
  suppressPackageStartupMessages({
    library(survival);  library(dplyr);  library(tidyr)
    library(caret)      # for nearZeroVar()
  })
  
  ## ------------------------------------------------------------------ ##
  ## 1.  Basic hygiene ------------------------------------------------- ##
  train <- train |> filter(timeDiff > 0)
  test  <- test  |> filter(timeDiff > 0)
  
  ## keep only requested features
  keep <- c("timeDiff", "status",
            if (is.null(features)) names(train) else features)
  train <- train |> select(any_of(unique(keep)))
  test  <-  test |> select(any_of(unique(keep)))
  
  ## ------------------------------------------------------------------ ##
  ## 2.  Align factor levels & drop empties --------------------------- ##
  common_factors <- names(train)[sapply(train, is.factor)]
  for (v in common_factors) {
    lev <- levels(train[[v]])
    train[[v]] <- factor(train[[v]], levels = lev) |> droplevels()
    test [[v]] <- factor(test [[v]], levels = lev)
  }
  
  ## ------------------------------------------------------------------ ##
  ## 3.  Remove zero / near‑zero variance predictors ------------------ ##
  drop_nzv <- nearZeroVar(train |> select(-timeDiff, -status),
                          freqCut = 1/(1 - nzv_cutoff),
                          uniqueCut = nzv_cutoff * 100,
                          names = TRUE)
  if (length(drop_nzv))
    message("Dropped near‑zero‑variance cols: ",
            paste0(drop_nzv, collapse = ", "))
  
  train <- train |> select(-all_of(drop_nzv))
  test  <-  test |>  select(-all_of(drop_nzv))
  
  
  ## ------------------------------------------------------------------ ##
  ## 5.  Fit with safe controls --------------------------------------- ##
  ctrl <- survreg.control(maxiter = maxiter,
                          rel.tolerance = rel.tol,
                          outer.max = 20)
  
  fmla <- as.formula(Surv(timeDiff / 86400, status) ~ .)
  aft  <- tryCatch(
    survreg(fmla, data = train, dist = dist, control = ctrl),
    error = identity, warning = identity
  )
  
  ## 5a.  If singularities: refit without NA coefficients ------------- ##
  if (inherits(aft, "survreg") && anyNA(coef(aft))) {
    bad  <- names(coef(aft))[is.na(coef(aft))]
    message("Refitting after dropping singular terms: ",
            paste(bad, collapse = ", "))
    keep <- setdiff(names(train), c(bad, "timeDiff", "status"))
    fmla <- reformulate(keep, response = "Surv(timeDiff / 86400, status)")
    aft  <- survreg(fmla, data = train, dist = dist, control = ctrl)
  }
  
  if (!inherits(aft, "survreg"))
    stop("survreg failed after refit: ", conditionMessage(aft))
  
  ## ------------------------------------------------------------------ ##
  ## 6.  Prediction ---------------------------------------------------- ##
  pred <- predict(aft, newdata = test) / 86400  # back to days
  list(prediction_days = pred, model = aft)
}
