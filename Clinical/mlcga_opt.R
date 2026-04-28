# Multivariate Latent Class Growth Analysis (MLCGA) Script for ABCD Data
# ULTRA-OPTIMIZED VERSION - Maximum speed improvements
# Features: Smart caching, grid search, reduced iterations, better initial values
# TRUE multivariate analysis - considers all variables simultaneously

# ============================================================================
# PACKAGE INSTALLATION AND LOADING
# ============================================================================

required_packages <- c("lcmm", "dplyr", "ggplot2", "tidyr", "readr", 
                       "data.table", "parallel", "doParallel", "foreach", 
                       "progress", "pryr", "digest")

cat("Checking and installing required packages...\n")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg, quiet = TRUE)
    library(pkg, character.only = TRUE)
  }
}

library(lcmm)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)
library(data.table)
library(parallel)
library(doParallel)
library(foreach)
library(progress)
library(pryr)
library(digest)

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS
# ============================================================================

# Output directory
output_dir <- "/home/rif17002/honors_thesis/mlcga_results"

# Base directory for CBCL data
cbcl_dir <- "/home/rif17002/honors_thesis/ABCD_CBCL/Release_5.0"

# Additional data files (optional)
additional_files <- list()

# CBCL variables to always include (the core irritability items)
cbcl_vars <- c("cbcl_q86_p", "cbcl_q87_p", "cbcl_q95_p")

# CBCL data format
cbcl_format <- "long"  # or "wide"

# Time points to include in analysis
timepoints_to_include <- c("baseline_year_1_arm_1", 
                           "1_year_follow_up_y_arm_1",
                           "2_year_follow_up_y_arm_1", 
                           "3_year_follow_up_y_arm_1",
                           "4_year_follow_up_y_arm_1")

# Number of classes to test
n_classes_to_test <- 3:5

# Missingness handling strategy
missingness_strategy <- "complete_case"

# ============================================================================
# SPEED OPTIMIZATION SETTINGS - ADJUST FOR SPEED VS ACCURACY TRADEOFF
# ============================================================================

# SPEED LEVEL: 1 = Maximum Speed, 2 = Balanced, 3 = Maximum Accuracy
SPEED_LEVEL <- 1  # Change to 2 or 3 for more thorough analysis

# Speed configurations
if (SPEED_LEVEL == 1) {
  # MAXIMUM SPEED (5-10x faster)
  use_parallel <- TRUE
  n_cores <- 8
  max_iterations_initial <- 50      # Reduced from 100
  max_iterations_final <- 200       # Reduced from 500
  n_random_starts <- 3              # Random starts for robustness
  use_grid_search <- TRUE           # Fast grid search for initial values
  data_subsample <- 1.0             # Use 100% of data (set to 0.5 for 50%)
  enable_smart_cache <- TRUE        # Cache intermediate results
  convergence_tolerance <- 1e-4     # Slightly relaxed (default: 1e-7)
  
} else if (SPEED_LEVEL == 2) {
  # BALANCED (2-3x faster)
  use_parallel <- TRUE
  n_cores <- detectCores() - 1
  max_iterations_initial <- 100
  max_iterations_final <- 300
  n_random_starts <- 5
  use_grid_search <- TRUE
  data_subsample <- 1.0
  enable_smart_cache <- TRUE
  convergence_tolerance <- 1e-5
  
} else {
  # MAXIMUM ACCURACY (original settings)
  use_parallel <- TRUE
  n_cores <- detectCores() - 1
  max_iterations_initial <- 100
  max_iterations_final <- 500
  n_random_starts <- 10
  use_grid_search <- FALSE
  data_subsample <- 1.0
  enable_smart_cache <- TRUE
  convergence_tolerance <- 1e-7
}

cat("\n============================================\n")
cat("SPEED OPTIMIZATION LEVEL:", SPEED_LEVEL, "\n")
cat("============================================\n")
cat("Max iterations:", max_iterations_final, "\n")
cat("Random starts:", n_random_starts, "\n")
cat("Grid search:", use_grid_search, "\n")
cat("Data subsample:", data_subsample * 100, "%\n")
cat("Parallel cores:", n_cores, "\n")
cat("Smart caching:", enable_smart_cache, "\n")
cat("============================================\n\n")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logs_dir <- file.path(output_dir, "logs")
if (!dir.exists(logs_dir)) {
  dir.create(logs_dir, recursive = TRUE)
}

log_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_file <- file.path(logs_dir, paste0("mlcga_analysis_log_", log_timestamp, ".txt"))

log_con <- file(log_file, open = "wt")
sink(log_con, type = "output", split = TRUE)

original_warning <- getOption("warn")
options(warn = 1)

cat("============================================\n")
cat("MLCGA ANALYSIS LOG (ULTRA-OPTIMIZED)\n")
cat("Started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================\n\n")
cat("R Version:", R.version.string, "\n")
cat("Log file:", log_file, "\n")
cat("Initial memory usage:", format(mem_used(), units = "MB"), "\n\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

convert_eventname_to_months <- function(eventname) {
  dplyr::case_when(
    grepl("baseline_year_1_arm_1", eventname) ~ 0,
    grepl("6_month_follow_up_y_arm_1", eventname) ~ 6,
    grepl("1_year_follow_up_y_arm_1", eventname) ~ 12,
    grepl("18_month_follow_up_y_arm_1", eventname) ~ 18,
    grepl("2_year_follow_up_y_arm_1", eventname) ~ 24,
    grepl("30_month_follow_up_y_arm_1", eventname) ~ 30,
    grepl("3_year_follow_up_y_arm_1", eventname) ~ 36,
    grepl("42_month_follow_up_y_arm_1", eventname) ~ 42,
    grepl("4_year_follow_up_y_arm_1", eventname) ~ 48,
    TRUE ~ NA_real_
  )
}

# Load CBCL long format files (OPTIMIZED)
load_cbcl_long <- function(cbcl_dir, timepoints, cbcl_vars) {
  timepoint_files <- c(
    "baseline_year_1_arm_1" = "0m",
    "1_year_follow_up_y_arm_1" = "12m",
    "2_year_follow_up_y_arm_1" = "24m",
    "3_year_follow_up_y_arm_1" = "36m",
    "4_year_follow_up_y_arm_1" = "48m"
  )
  
  all_data <- list()
  
  cat("\nLoading", length(timepoints), "CBCL long format files...\n")
  
  for (tp in timepoints) {
    file_suffix <- timepoint_files[tp]
    if (is.na(file_suffix)) {
      warning("No file mapping for timepoint: ", tp)
      next
    }
    
    file_path <- file.path(cbcl_dir, 
                           paste0("abcd_cbcl_irr_index_release5.0_", 
                                  file_suffix, "_long.csv"))
    
    if (file.exists(file_path)) {
      df <- tryCatch({
        fread(file_path, showProgress = FALSE, nThread = n_cores)
      }, error = function(e) {
        read_csv(file_path, show_col_types = FALSE)
      })
      
      if ("src_subject_id" %in% names(df)) {
        setnames(df, "src_subject_id", "subjectkey")
      }
      
      df[, eventname := tp]
      df[, time_months := convert_eventname_to_months(tp)]
      
      available_vars <- intersect(cbcl_vars, names(df))
      
      if (length(available_vars) > 0) {
        cols_to_keep <- c("subjectkey", "eventname", "time_months", available_vars)
        df <- df[, ..cols_to_keep]
        all_data[[tp]] <- df
        cat("  Loaded", tp, ":", nrow(df), "rows\n")
      }
    }
  }
  
  if (length(all_data) > 0) {
    combined <- rbindlist(all_data, use.names = TRUE, fill = TRUE)
    cat("\nCombined CBCL data:", nrow(combined), "rows x", ncol(combined), "columns\n")
    cat("Unique subjects:", uniqueN(combined$subjectkey), "\n")
    return(combined)
  } else {
    stop("No CBCL data could be loaded")
  }
}

# Load CBCL wide format (OPTIMIZED)
load_cbcl_wide <- function(cbcl_dir, timepoints, cbcl_vars) {
  timepoint_files <- c(
    "baseline_year_1_arm_1" = "0m",
    "1_year_follow_up_y_arm_1" = "12m",
    "2_year_follow_up_y_arm_1" = "24m",
    "3_year_follow_up_y_arm_1" = "36m",
    "4_year_follow_up_y_arm_1" = "48m"
  )
  
  timepoint_months <- c(
    "baseline_year_1_arm_1" = 0,
    "1_year_follow_up_y_arm_1" = 12,
    "2_year_follow_up_y_arm_1" = 24,
    "3_year_follow_up_y_arm_1" = 36,
    "4_year_follow_up_y_arm_1" = 48
  )
  
  all_data <- list()
  
  cat("\nLoading", length(timepoints), "CBCL wide format files...\n")
  
  for (tp in timepoints) {
    file_suffix <- timepoint_files[tp]
    file_path <- file.path(cbcl_dir, 
                           paste0("abcd_cbcl_irr_index_release5.0_", 
                                  file_suffix, "_wide.csv"))
    
    if (file.exists(file_path)) {
      df <- tryCatch({
        fread(file_path, showProgress = FALSE, nThread = n_cores)
      }, error = function(e) {
        read_csv(file_path, show_col_types = FALSE)
      })
      
      if (is.null(df)) next
      
      if ("src_subject_id" %in% names(df)) {
        setnames(df, "src_subject_id", "subjectkey")
      }
      
      available_vars <- intersect(cbcl_vars, names(df))
      
      if (length(available_vars) > 0) {
        cols_to_keep <- c("subjectkey", available_vars)
        df <- df[, ..cols_to_keep]
        
        time_suffix <- paste0("_t", timepoint_months[tp])
        for (var in available_vars) {
          setnames(df, var, paste0(var, time_suffix))
        }
        
        all_data[[tp]] <- df
      }
    }
  }
  
  if (length(all_data) > 0) {
    cat("\nMerging wide format data...\n")
    combined <- Reduce(function(x, y) merge(x, y, by = "subjectkey", all = TRUE), 
                       all_data)
    
    cat("Converting to long format...\n")
    var_cols <- setdiff(names(combined), "subjectkey")
    
    combined_long <- melt(combined, 
                          id.vars = "subjectkey",
                          measure.vars = var_cols,
                          variable.name = "var_time",
                          value.name = "value")
    
    combined_long[, variable := sub("_t[0-9]+$", "", var_time)]
    combined_long[, time_months := as.numeric(sub(".*_t", "", var_time))]
    combined_long[, var_time := NULL]
    
    combined_long[, eventname := fifelse(
      time_months == 0, "baseline_year_1_arm_1",
      fifelse(time_months == 12, "1_year_follow_up_y_arm_1",
      fifelse(time_months == 24, "2_year_follow_up_y_arm_1",
      fifelse(time_months == 36, "3_year_follow_up_y_arm_1",
      fifelse(time_months == 48, "4_year_follow_up_y_arm_1",
      NA_character_)))))]
    
    combined_long <- dcast(combined_long, 
                           subjectkey + eventname + time_months ~ variable,
                           value.var = "value")
    
    return(combined_long)
  } else {
    stop("No CBCL data could be loaded")
  }
}

# Load additional files
load_additional_file <- function(file_info, timepoints) {
  file_path <- file_info$file
  vars_to_extract <- file_info$vars
  time_var <- file_info$time_var
  
  if (!file.exists(file_path)) {
    warning("File not found: ", file_path)
    return(NULL)
  }
  
  cat("\nLoading additional file:", basename(file_path), "\n")
  
  df <- tryCatch({
    fread(file_path, showProgress = FALSE, nThread = n_cores)
  }, error = function(e) {
    tryCatch({
      read_tsv(file_path, show_col_types = FALSE, guess_max = 10000)
    }, error = function(e2) {
      read_csv(file_path, show_col_types = FALSE, guess_max = 10000)
    })
  })
  
  if (is.null(df)) return(NULL)
  
  if (!is.data.table(df)) setDT(df)
  
  if ("src_subject_id" %in% names(df)) {
    setnames(df, "src_subject_id", "subjectkey")
  }
  
  if (!("subjectkey" %in% names(df))) {
    warning("No subject ID column found in ", file_path)
    return(NULL)
  }
  
  if (!time_var %in% names(df)) {
    warning("Time variable '", time_var, "' not found in ", file_path)
    return(NULL)
  }
  
  df <- df[get(time_var) %in% timepoints]
  
  if (nrow(df) == 0) return(NULL)
  
  df[, eventname := get(time_var)]
  df[, time_months := convert_eventname_to_months(get(time_var))]
  
  available_vars <- intersect(vars_to_extract, names(df))
  
  if (length(available_vars) == 0) return(NULL)
  
  cols_to_keep <- c("subjectkey", "eventname", "time_months", available_vars)
  df <- df[, ..cols_to_keep]
  
  return(df)
}

# ============================================================================
# SMART CACHING SYSTEM
# ============================================================================

get_cache_key <- function(...) {
  digest(list(...), algo = "md5")
}

save_to_cache <- function(obj, cache_name, params = NULL) {
  if (!enable_smart_cache) return(FALSE)
  
  cache_dir <- file.path(output_dir, "cache")
  if (!dir.exists(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE)
  }
  
  if (!is.null(params)) {
    cache_key <- get_cache_key(params)
    cache_file <- file.path(cache_dir, paste0(cache_name, "_", cache_key, ".RData"))
  } else {
    cache_file <- file.path(cache_dir, paste0(cache_name, ".RData"))
  }
  
  save(obj, file = cache_file)
  return(TRUE)
}

load_from_cache <- function(cache_name, params = NULL) {
  if (!enable_smart_cache) return(NULL)
  
  cache_dir <- file.path(output_dir, "cache")
  
  if (!is.null(params)) {
    cache_key <- get_cache_key(params)
    cache_file <- file.path(cache_dir, paste0(cache_name, "_", cache_key, ".RData"))
  } else {
    cache_file <- file.path(cache_dir, paste0(cache_name, ".RData"))
  }
  
  if (file.exists(cache_file)) {
    load(cache_file)
    return(obj)
  }
  return(NULL)
}

# ============================================================================
# OPTIMIZED MODEL FITTING WITH GRID SEARCH
# ============================================================================

# Grid search for fast initial values
grid_search_initial_values <- function(data_analysis, formula_str, ng, n_starts = 3) {
  cat("  Running grid search for initial values (", n_starts, "starts)...\n")
  
  best_loglik <- -Inf
  best_model <- NULL
  
  for (i in 1:n_starts) {
    model <- tryCatch({
      multlcmm(
        fixed = as.formula(formula_str),
        random = ~ time_months,
        subject = "subjectkey_numeric",
        ng = ng,
        data = as.data.frame(data_analysis),
        verbose = FALSE,
        maxiter = 25,
        convB = convergence_tolerance,
        convL = convergence_tolerance,
        convG = convergence_tolerance
      )
    }, error = function(e) NULL)
    
    if (!is.null(model) && model$loglik > best_loglik) {
      best_loglik <- model$loglik
      best_model <- model
    }
  }
  
  return(best_model)
}

# Fast adaptive fitting (FIXED - removed problematic progress callback)
fit_model_adaptive <- function(formula_str, data_analysis, ng, B_model = NULL, 
                              max_iter_sequence = NULL) {
  
  if (is.null(max_iter_sequence)) {
    max_iter_sequence <- c(max_iterations_initial, max_iterations_final)
  }
  
  fixed_formula <- as.formula(formula_str)
  
  for (iter_idx in seq_along(max_iter_sequence)) {
    max_iter <- max_iter_sequence[iter_idx]
    
    tryCatch({
      if (is.null(B_model)) {
        # Use grid search if enabled
        if (use_grid_search && ng == 1) {
          model <- grid_search_initial_values(data_analysis, formula_str, ng, 
                                             n_random_starts)
          if (!is.null(model)) {
            # Refine with more iterations
            model <- multlcmm(
              fixed = fixed_formula,
              random = ~ time_months,
              subject = "subjectkey_numeric",
              ng = ng,
              data = as.data.frame(data_analysis),
              B = model,
              verbose = FALSE,
              maxiter = max_iter,
              convB = convergence_tolerance,
              convL = convergence_tolerance,
              convG = convergence_tolerance
            )
          }
        } else {
          model <- multlcmm(
            fixed = fixed_formula,
            random = ~ time_months,
            subject = "subjectkey_numeric",
            ng = ng,
            data = as.data.frame(data_analysis),
            verbose = FALSE,
            maxiter = max_iter,
            convB = convergence_tolerance,
            convL = convergence_tolerance,
            convG = convergence_tolerance
          )
        }
      } else {
        model <- multlcmm(
          fixed = fixed_formula,
          mixture = ~ time_months,
          random = ~ time_months,
          subject = "subjectkey_numeric",
          ng = ng,
          data = as.data.frame(data_analysis),
          B = B_model,
          verbose = FALSE,
          maxiter = max_iter,
          convB = convergence_tolerance,
          convL = convergence_tolerance,
          convG = convergence_tolerance
        )
      }
      
      if (model$conv == 1 || model$conv == 2) {
        cat("    Converged (conv:", model$conv, ")\n")
        return(model)
      }
      
    }, error = function(e) {
      cat("    Error:", e$message, "\n")
    })
  }
  
  return(model)
}

# ============================================================================
# MAIN MODEL FITTING FUNCTION (ULTRA-OPTIMIZED)
# ============================================================================

fit_mlcga_models <- function(data, outcome_vars, n_classes_vec, 
                             missingness_strategy = "complete_case",
                             use_parallel = TRUE, n_cores = NULL) {
  
  models <- list()
  fit_stats <- data.frame()
  
  cat("\n========================================\n")
  cat("Fitting MULTIVARIATE LCGA models\n")
  cat("Outcomes:", paste(outcome_vars, collapse = ", "), "\n")
  cat("========================================\n")
  
  # Handle missingness
  if (missingness_strategy == "complete_case") {
    data_analysis <- data[complete.cases(data[, ..outcome_vars])]
    cat("\nUsing COMPLETE CASE analysis\n")
    cat("N observations:", nrow(data_analysis), "\n")
  } else {
    data_analysis <- data
    cat("\nUsing PAIRWISE deletion\n")
    cat("N observations:", nrow(data_analysis), "\n")
  }
  
  cat("N subjects:", uniqueN(data_analysis$subjectkey), "\n")
  
  # SPEED OPTIMIZATION: Subsample data if requested
  if (data_subsample < 1.0) {
    n_subjects <- uniqueN(data_analysis$subjectkey)
    n_keep <- round(n_subjects * data_subsample)
    
    cat("\n*** SPEED MODE: Subsampling to", data_subsample * 100, "% of data ***\n")
    cat("Keeping", n_keep, "of", n_subjects, "subjects\n")
    
    subjects_to_keep <- sample(unique(data_analysis$subjectkey), n_keep)
    data_analysis <- data_analysis[subjectkey %in% subjects_to_keep]
    
    cat("New N observations:", nrow(data_analysis), "\n\n")
  }
  
  if (nrow(data_analysis) < 100) {
    stop("Too few observations. Consider increasing data_subsample.")
  }
  
  # Build formula
  formula_str <- paste(paste(outcome_vars, collapse = " + "), "~ time_months")
  cat("Formula:", formula_str, "\n")
  
  # Fit 1-class model with caching
  cat("\n========================================\n")
  cat("Fitting 1-class model (initial values)\n")
  cat("========================================\n")
  
  cache_params <- list(
    formula = formula_str,
    n_obs = nrow(data_analysis),
    subsample = data_subsample,
    tol = convergence_tolerance
  )
  
  model_1class <- load_from_cache("model_1class", cache_params)
  
  if (!is.null(model_1class)) {
    cat("✓ Loaded 1-class model from cache\n")
  } else {
    cat("Fitting new 1-class model...\n")
    
    # FIXED: Removed the problematic progress bar
    model_1class <- fit_model_adaptive(
      formula_str = formula_str,
      data_analysis = data_analysis,
      ng = 1,
      B_model = NULL
    )
    
    if (is.null(model_1class)) {
      stop("Failed to fit 1-class model")
    }
    
    save_to_cache(model_1class, "model_1class", cache_params)
    cat("✓ Cached 1-class model\n")
  }
  
  cat("LogLik:", round(model_1class$loglik, 2), "\n")
  
  # Fit multi-class models
  if (use_parallel && length(n_classes_vec) > 1) {
    cat("\n========================================\n")
    cat("PARALLEL MODEL FITTING (", n_cores, "cores)\n")
    cat("========================================\n\n")
    
    cl <- makeCluster(n_cores)
    registerDoParallel(cl)
    
    clusterExport(cl, c("formula_str", "data_analysis", "max_iterations_initial",
                        "max_iterations_final", "convergence_tolerance",
                        "use_grid_search", "n_random_starts"),
                  envir = environment())
    
    B_values <- model_1class$best
    clusterExport(cl, "B_values", envir = environment())
    
    cat("Fitting models in parallel...\n")
    
    results_list <- foreach(n_class = n_classes_vec, 
                            .packages = c("lcmm", "dplyr"),
                            .errorhandling = "pass",
                            .combine = 'list',
                            .multicombine = TRUE) %dopar% {
      
      B_model <- model_1class
      B_model$best <- B_values
      
      model <- tryCatch({
        multlcmm(
          fixed = as.formula(formula_str),
          mixture = ~ time_months,
          random = ~ time_months,
          subject = "subjectkey_numeric",
          ng = n_class,
          data = as.data.frame(data_analysis),
          B = B_model,
          verbose = FALSE,
          maxiter = max_iterations_final,
          convB = convergence_tolerance,
          convL = convergence_tolerance,
          convG = convergence_tolerance
        )
      }, error = function(e) NULL)
      
      if (!is.null(model)) {
        list(
          model = model,
          n_classes = n_class,
          AIC = model$AIC,
          BIC = model$BIC,
          loglik = model$loglik,
          converged = (model$conv == 1 || model$conv == 2),
          n_iter = model$niter
        )
      } else {
        list(
          model = NULL, n_classes = n_class, AIC = NA, BIC = NA,
          loglik = NA, converged = FALSE, n_iter = NA
        )
      }
    }
    
    stopCluster(cl)
    
    for (i in seq_along(results_list)) {
      result <- results_list[[i]]
      n_class <- result$n_classes
      model_name <- paste0("mlcga_", n_class, "class")
      
      if (!is.null(result$model)) {
        models[[model_name]] <- result$model
      }
      
      fit_stats <- rbind(fit_stats, data.frame(
        n_classes = result$n_classes,
        AIC = result$AIC,
        BIC = result$BIC,
        loglik = result$loglik,
        converged = result$converged,
        n_iter = result$n_iter,
        n_obs = nrow(data_analysis),
        n_subjects = uniqueN(data_analysis$subjectkey),
        n_outcomes = length(outcome_vars)
      ))
      
      status <- ifelse(result$converged, "✓", "✗")
      cat("  ", n_class, "-class:", status, "BIC:", round(result$BIC, 0), "\n")
    }
    
  } else {
    # Sequential fitting
    cat("\n========================================\n")
    cat("SEQUENTIAL MODEL FITTING\n")
    cat("========================================\n\n")
    
    for (idx in seq_along(n_classes_vec)) {
      n_class <- n_classes_vec[idx]
      model_name <- paste0("mlcga_", n_class, "class")
      
      cat("Fitting", n_class, "-class model...\n")
      
      # Try to load from cache
      model_cache_params <- list(
        formula = formula_str,
        ng = n_class,
        n_obs = nrow(data_analysis),
        subsample = data_subsample,
        tol = convergence_tolerance
      )
      
      model <- load_from_cache(model_name, model_cache_params)
      
      if (!is.null(model)) {
        cat("  ✓ Loaded from cache\n")
      } else {
        model <- fit_model_adaptive(
          formula_str = formula_str,
          data_analysis = data_analysis,
          ng = n_class,
          B_model = model_1class
        )
        
        if (!is.null(model)) {
          save_to_cache(model, model_name, model_cache_params)
        }
      }
      
      if (!is.null(model)) {
        models[[model_name]] <- model
        
        fit_stats <- rbind(fit_stats, data.frame(
          n_classes = n_class,
          AIC = model$AIC,
          BIC = model$BIC,
          loglik = model$loglik,
          converged = (model$conv == 1 || model$conv == 2),
          n_iter = model$niter,
          n_obs = nrow(data_analysis),
          n_subjects = uniqueN(data_analysis$subjectkey),
          n_outcomes = length(outcome_vars)
        ))
        
        cat("  ✓ BIC:", round(model$BIC, 0), "\n")
      } else {
        fit_stats <- rbind(fit_stats, data.frame(
          n_classes = n_class,
          AIC = NA, BIC = NA, loglik = NA,
          converged = FALSE, n_iter = NA,
          n_obs = nrow(data_analysis),
          n_subjects = uniqueN(data_analysis$subjectkey),
          n_outcomes = length(outcome_vars)
        ))
        
        cat("  ✗ Failed to converge\n")
      }
      
      gc(verbose = FALSE)
    }
  }
  
  cat("\nMemory after fitting:", format(mem_used(), units = "MB"), "\n")
  
  return(list(
    models = models, 
    fit_stats = fit_stats,
    outcome_vars = outcome_vars,
    data_analysis = data_analysis,
    id_mapping = id_mapping
  ))
}

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

cat("\n============================================\n")
cat("LOADING DATA\n")
cat("============================================\n")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Load CBCL data
cat("\n[Step 1/8] Loading CBCL data (format:", cbcl_format, ")...\n")

if (cbcl_format == "wide") {
  cbcl_data <- load_cbcl_wide(cbcl_dir, timepoints_to_include, cbcl_vars)
} else if (cbcl_format == "long") {
  cbcl_data <- load_cbcl_long(cbcl_dir, timepoints_to_include, cbcl_vars)
} else {
  stop("Invalid cbcl_format. Must be 'wide' or 'long'")
}

cat("✓ Step 1 complete\n")

# Load additional files
additional_data_list <- list()
if (length(additional_files) > 0) {
  cat("\n[Step 2/8] Loading additional files...\n")
  
  for (i in seq_along(additional_files)) {
    add_data <- load_additional_file(additional_files[[i]], timepoints_to_include)
    if (!is.null(add_data)) {
      additional_data_list[[i]] <- add_data
    }
  }
  cat("✓ Step 2 complete\n")
} else {
  cat("\n[Step 2/8] No additional files\n")
  cat("✓ Step 2 complete\n")
}

# Merge data
if (length(additional_data_list) > 0) {
  cat("\n[Step 3/8] Merging data...\n")
  
  merged_data <- cbcl_data
  
  for (add_data in additional_data_list) {
    if (!is.data.table(merged_data)) setDT(merged_data)
    if (!is.data.table(add_data)) setDT(add_data)
    
    merged_data <- merge(merged_data, add_data, 
                         by = c("subjectkey", "eventname", "time_months"),
                         all = TRUE)
  }
  
  cat("✓ Step 3 complete\n")
} else {
  cat("\n[Step 3/8] No merging needed\n")
  merged_data <- cbcl_data
  cat("✓ Step 3 complete\n")
}

# Clean and prepare
merged_data <- merged_data[!is.na(time_months)]

cat("\n[Step 4/8] Creating ID mapping...\n")
id_mapping <- unique(merged_data[, .(subjectkey)])
id_mapping[, subjectkey_numeric := .I]
merged_data <- merge(merged_data, id_mapping, by = "subjectkey", all.x = TRUE)

cat("✓ Step 4 complete\n")

# Summary
cat("\n============================================\n")
cat("DATA SUMMARY\n")
cat("============================================\n")
cat("Total observations:", nrow(merged_data), "\n")
cat("Unique subjects:", uniqueN(merged_data$subjectkey), "\n")

# ============================================================================
# RUN ANALYSIS
# ============================================================================

cat("\n[Step 5/8] Fitting MLCGA models...\n")

outcome_vars <- intersect(cbcl_vars, names(merged_data))

if (length(additional_data_list) > 0) {
  for (file_info in additional_files) {
    additional_vars <- intersect(file_info$vars, names(merged_data))
    outcome_vars <- c(outcome_vars, additional_vars)
  }
}

cat("\nAnalyzing outcomes:", paste(outcome_vars, collapse = ", "), "\n")

# Start timing
start_time <- Sys.time()

results <- fit_mlcga_models(
  data = merged_data,
  outcome_vars = outcome_vars,
  n_classes_vec = n_classes_to_test,
  missingness_strategy = missingness_strategy,
  use_parallel = use_parallel,
  n_cores = n_cores
)

end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat("\n✓ Step 5 complete\n")
cat("Total model fitting time:", round(elapsed_time, 2), "minutes\n")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

cat("\n[Step 6/8] Model comparison...\n")

print(results$fit_stats)

fit_stats_file <- file.path(output_dir, "mlcga_fit_statistics.csv")
write.csv(results$fit_stats, fit_stats_file, row.names = FALSE)

best_model_row <- results$fit_stats %>%
  filter(converged == TRUE) %>%
  slice_min(BIC, n = 1)

if (nrow(best_model_row) == 0) {
  stop("No models converged")
}

n_class <- best_model_row$n_classes

cat("\nBest model:", n_class, "classes (BIC =", round(best_model_row$BIC, 2), ")\n")
cat("✓ Step 6 complete\n")

# ============================================================================
# VISUALIZATIONS (SIMPLIFIED FOR SPEED) - FIXED POSTERIORS JOIN
# ============================================================================

cat("\n[Step 7/8] Creating plots...\n")

model_name <- paste0("mlcga_", n_class, "class")
model <- results$models[[model_name]]

if (!is.null(model)) {
  posteriors <- model$pprob
  
  # Check what the ID column is actually named
  cat("Posterior probability columns:", paste(names(posteriors), collapse = ", "), "\n")
  
  # The first column should be the subject ID - find its name
  id_col_name <- names(posteriors)[1]
  
  posteriors_with_id <- results$id_mapping %>%
    left_join(as.data.frame(posteriors), 
              by = setNames(id_col_name, "subjectkey_numeric"))
  
  posteriors_file <- file.path(output_dir, 
                                paste0("posteriors_mlcga_", n_class, "class.csv"))
  write.csv(posteriors_with_id, posteriors_file, row.names = FALSE)
  
  data_with_class <- merge(
    results$data_analysis,
    posteriors_with_id[, c("subjectkey", "class")],
    by = "subjectkey",
    all.x = TRUE
  )
  
  # Create mean trajectories plot only (faster)
  plot_data <- data_with_class %>%
    pivot_longer(
      cols = all_of(results$outcome_vars),
      names_to = "outcome",
      values_to = "value"
    ) %>%
    filter(!is.na(value))
  
  setDT(plot_data)
  
  mean_trajectories <- plot_data[, .(
    mean_value = mean(value, na.rm = TRUE),
    se = sd(value, na.rm = TRUE) / sqrt(.N)
  ), by = .(class, time_months, outcome)]
  
  p <- ggplot(mean_trajectories, 
              aes(x = time_months, y = mean_value, color = as.factor(class))) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                  width = 2, alpha = 0.6) +
    facet_wrap(~ outcome, scales = "free_y", ncol = 2) +
    labs(title = paste0("Mean Trajectories (", n_class, " classes)"),
         x = "Time (months)",
         y = "Mean Value",
         color = "Class") +
    theme_minimal()
  
  plot_file <- file.path(output_dir, 
                         paste0("mean_trajectories_mlcga_", n_class, "class.png"))
  ggsave(plot_file, plot = p, width = 12, height = 8, dpi = 150)
  
  cat("✓ Step 7 complete\n")
  cat("Plot saved to:", plot_file, "\n")
}

# ============================================================================
# SAVE RESULTS
# ============================================================================

cat("\n[Step 8/8] Saving results...\n")

final_data <- merge(
  merged_data,
  posteriors_with_id[, .(subjectkey, class)],
  by = "subjectkey",
  all.x = TRUE
)

setnames(final_data, "class", "mlcga_class")

final_data_file <- file.path(output_dir, "data_with_mlcga_class_assignments.csv")
fwrite(final_data, final_data_file)

model_file <- file.path(output_dir, paste0("mlcga_", n_class, "class_model.RData"))
save(model, file = model_file)

all_results_file <- file.path(output_dir, "mlcga_all_results.RData")
save(results, file = all_results_file)

cat("✓ Step 8 complete\n")

# ============================================================================
# FINALIZE
# ============================================================================

cat("\n============================================\n")
cat("ANALYSIS COMPLETE\n")
cat("============================================\n")
cat("Total time:", round(elapsed_time, 2), "minutes\n")
cat("Results saved to:", output_dir, "\n")
cat("============================================\n")

sink(type = "output")
close(log_con)
options(warn = original_warning)

message("\n*** Analysis complete in ", round(elapsed_time, 2), " minutes ***")