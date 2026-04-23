# Multivariate Latent Class Growth Analysis (MLCGA) Script for ABCD Data
# TRUE multivariate analysis - considers all variables simultaneously
# Handles ABCD-specific time variables and CBCL irritability data

# Install required packages if not already installed
if (!require("lcmm")) install.packages("lcmm")
if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("tidyr")) install.packages("tidyr")
if (!require("readr")) install.packages("readr")

library(lcmm)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS
# ============================================================================

# Output directory
output_dir <- "/home/rif17002/honors_thesis/mlcga_results"

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Create logs directory if it doesn't exist
logs_dir <- file.path(output_dir, "logs")
if (!dir.exists(logs_dir)) {
  dir.create(logs_dir, recursive = TRUE)
}

# Create log file with timestamp in the logs directory
log_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_file <- file.path(logs_dir, paste0("mlcga_analysis_log_", log_timestamp, ".txt"))

# Open log file connection
log_con <- file(log_file, open = "wt")

# Redirect stdout to both console and log file (split = TRUE keeps console output)
sink(log_con, type = "output", split = TRUE)

# For messages and warnings, we'll use a custom handler since split doesn't work for messages
# Save the original warning handler
original_warning <- getOption("warn")
options(warn = 1)  # Print warnings as they occur

# Log session info at start
cat("============================================\n")
cat("MLCGA ANALYSIS LOG\n")
cat("Started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================\n\n")
cat("R Version:\n")
print(R.version.string)
cat("\nLoaded Packages:\n")
print(sessionInfo())
cat("\n")
cat("Log file:", log_file, "\n\n")

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS
# ============================================================================

# Base directory for CBCL data
cbcl_dir <- "/home/rif17002/honors_thesis/ABCD_CBCL/Release_5.0"

# Additional data files (optional) - other variables you want to include
# Each element should be a list with 'file' path and 'vars' to extract
additional_files <- list(
  # ODD
  # list(
  #   file = "/home/rif17002/honors_thesis/ABCD_files/opp_defiant_disorder_p01.txt",
  #   vars = c("ksads_odd_raw_1020_p"),
  #   time_var = "eventname"
  # ),

  # list(
  #   file = "/home/rif17002/honors_thesis/ABCD_files/diff_emotion_reg_p01.txt",
  #   vars = c("ders_emotion_overwhelm_p", "ders_upset_behavior_p"),
  #   time_var = "eventname"
  # )
)

# CBCL variables to always include (the core irritability items)
cbcl_vars <- c("cbcl_q86_p", "cbcl_q87_p", "cbcl_q95_p")

# Time points to include in analysis
# Will be converted to numeric time variable (0, 12, 24, 36, 48 months)
timepoints_to_include <- c("baseline_year_1_arm_1", 
                           "1_year_follow_up_y_arm_1",
                           "2_year_follow_up_y_arm_1", 
                           "3_year_follow_up_y_arm_1",
                           "4_year_follow_up_y_arm_1")

# Number of classes to test
n_classes_to_test <- 3:5

# Missingness handling strategy
# "complete_case" = only subjects with complete data on ALL outcomes at ALL timepoints
# "pairwise" = use all available data (allows missing, but requires multlcmm to handle it)
missingness_strategy <- "complete_case"  

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Convert ABCD eventname to numeric months
convert_eventname_to_months <- function(eventname) {
  months <- case_when(
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
  return(months)
}

# Load and process CBCL long format files
load_cbcl_long <- function(cbcl_dir, timepoints, cbcl_vars) {
  # Map timepoints to file suffixes
  timepoint_files <- c(
    "baseline_year_1_arm_1" = "0m",
    "1_year_follow_up_y_arm_1" = "12m",
    "2_year_follow_up_y_arm_1" = "24m",
    "3_year_follow_up_y_arm_1" = "36m",
    "4_year_follow_up_y_arm_1" = "48m"
  )
  
  all_data <- list()
  
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
      cat("Loading:", file_path, "\n")
      
      df <- read_csv(file_path, show_col_types = FALSE)
      
      # Check for required columns
      if ("subjectkey" %in% names(df) || "src_subject_id" %in% names(df)) {
        # Standardize ID column name
        if ("src_subject_id" %in% names(df)) {
          df <- df %>% rename(subjectkey = src_subject_id)
        }
        
        # Add time information
        df <- df %>%
          mutate(
            eventname = tp,
            time_months = convert_eventname_to_months(tp)
          )
        
        # Select relevant columns
        id_col <- "subjectkey"
        available_vars <- intersect(cbcl_vars, names(df))
        
        if (length(available_vars) > 0) {
          df <- df %>%
            select(all_of(c(id_col, "eventname", "time_months", available_vars)))
          
          all_data[[tp]] <- df
          cat("  Loaded", nrow(df), "rows with", length(available_vars), "variables\n")
        } else {
          warning("  No CBCL variables found in ", file_path)
        }
      }
    } else {
      warning("File not found: ", file_path)
    }
  }
  
  if (length(all_data) > 0) {
    combined <- bind_rows(all_data)
    cat("\nCombined CBCL data:", nrow(combined), "rows x", ncol(combined), "columns\n")
    cat("Unique subjects:", length(unique(combined$subjectkey)), "\n")
    return(combined)
  } else {
    stop("No CBCL data could be loaded")
  }
}

# Load additional variable files
load_additional_file <- function(file_info, timepoints) {
  file_path <- file_info$file
  vars_to_extract <- file_info$vars
  time_var <- file_info$time_var
  
  if (!file.exists(file_path)) {
    warning("File not found: ", file_path)
    return(NULL)
  }
  
  cat("\nLoading additional file:", file_path, "\n")
  
  # Try to read as TSV first (ABCD .txt files are tab-delimited), then CSV
  df <- tryCatch({
    read_tsv(file_path, show_col_types = FALSE, guess_max = 10000)
  }, error = function(e) {
    cat("  TSV read failed, trying CSV...\n")
    read_csv(file_path, show_col_types = FALSE, guess_max = 10000)
  })
  
  cat("  File loaded successfully:", nrow(df), "rows,", ncol(df), "columns\n")
  cat("  First few column names:", paste(names(df)[1:min(10, ncol(df))], collapse = ", "), "\n")
  
  # Identify ID column - check if subjectkey exists
  if ("subjectkey" %in% names(df)) {
    cat("  Found 'subjectkey' column\n")
    id_col <- "subjectkey"
  } else if ("src_subject_id" %in% names(df)) {
    cat("  Found 'src_subject_id' column, renaming to 'subjectkey'\n")
    df <- df %>% rename(subjectkey = src_subject_id)
    id_col <- "subjectkey"
  } else {
    # Print ALL column names to help debug
    cat("  ERROR: No subject ID column found!\n")
    cat("  All columns:", paste(names(df), collapse = ", "), "\n")
    warning("No subject ID column found in ", file_path)
    return(NULL)
  }
  
  # Check for time variable
  if (!time_var %in% names(df)) {
    warning("Time variable '", time_var, "' not found in ", file_path)
    return(NULL)
  }
  
  # Filter to relevant timepoints
  df <- df %>%
    filter(!!sym(time_var) %in% timepoints) %>%
    mutate(
      eventname = !!sym(time_var),
      time_months = convert_eventname_to_months(!!sym(time_var))
    )
  
  # Select relevant variables
  available_vars <- intersect(vars_to_extract, names(df))
  
  if (length(available_vars) == 0) {
    warning("None of the requested variables found in ", file_path)
    return(NULL)
  }
  
  df <- df %>%
    select(subjectkey, eventname, time_months, all_of(available_vars))
  
  cat("  Loaded", nrow(df), "rows with variables:", paste(available_vars, collapse = ", "), "\n")
  
  return(df)
}

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

cat("\n============================================\n")
cat("LOADING AND PREPARING DATA\n")
cat("============================================\n")

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

# Load CBCL data
cat("\nLoading CBCL data...\n")
cbcl_data <- load_cbcl_long(cbcl_dir, timepoints_to_include, cbcl_vars)

# Load additional files if specified
additional_data_list <- list()
if (length(additional_files) > 0) {
  cat("\nLoading additional data files...\n")
  for (i in seq_along(additional_files)) {
    add_data <- load_additional_file(additional_files[[i]], timepoints_to_include)
    if (!is.null(add_data)) {
      additional_data_list[[i]] <- add_data
    }
  }
}

# Merge all data sources
if (length(additional_data_list) > 0) {
  cat("\nMerging data sources...\n")
  
  merged_data <- cbcl_data
  
  for (add_data in additional_data_list) {
    merged_data <- merged_data %>%
      full_join(add_data, by = c("subjectkey", "eventname", "time_months"))
  }
  
  cat("Merged data dimensions:", nrow(merged_data), "rows x", ncol(merged_data), "columns\n")
} else {
  merged_data <- cbcl_data
}

# Remove rows with missing time
merged_data <- merged_data %>%
  filter(!is.na(time_months))

# CRITICAL: Convert subjectkey to numeric for multlcmm
merged_data <- merged_data %>%
  mutate(subjectkey_numeric = as.numeric(as.factor(subjectkey)))

cat("\nConverted", length(unique(merged_data$subjectkey)), "subject IDs to numeric format\n")

# Summary statistics
cat("\n============================================\n")
cat("DATA SUMMARY\n")
cat("============================================\n")
cat("Total observations:", nrow(merged_data), "\n")
cat("Unique subjects:", length(unique(merged_data$subjectkey)), "\n")
cat("Time points:\n")
print(table(merged_data$eventname))
cat("\nVariables in dataset:\n")
print(names(merged_data))

# Check missingness
cat("\nMissing data summary:\n")
missing_summary <- merged_data %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  mutate(pct_missing = round(n_missing / nrow(merged_data) * 100, 2)) %>%
  arrange(desc(n_missing))

print(missing_summary)

# ============================================================================
# MODEL FITTING FUNCTIONS
# ============================================================================

fit_mlcga_models <- function(data, outcome_vars, n_classes_vec, missingness_strategy = "complete_case") {
  
  models <- list()
  fit_stats <- data.frame()
  
  cat("\n========================================\n")
  cat("Fitting MULTIVARIATE LCGA models\n")
  cat("Outcomes:", paste(outcome_vars, collapse = ", "), "\n")
  cat("========================================\n")
  
  # Handle missingness according to strategy
  if (missingness_strategy == "complete_case") {
    # Filter data to complete cases for all outcomes
    data_analysis <- data %>%
      filter(if_all(all_of(outcome_vars), ~ !is.na(.)))
    
    cat("\nUsing COMPLETE CASE analysis\n")
    cat("N observations with complete data across all outcomes:", nrow(data_analysis), "\n")
  } else {
    # Use all available data (pairwise deletion)
    data_analysis <- data
    cat("\nUsing PAIRWISE deletion (all available data)\n")
    cat("N observations:", nrow(data_analysis), "\n")
  }
  
  cat("N subjects:", length(unique(data_analysis$subjectkey)), "\n")
  
  if (nrow(data_analysis) < 100) {
    stop("Too few observations with data. Consider changing missingness strategy or checking data.")
  }
  
  # First, fit a 1-class model to get initial values
  cat("\nFitting 1-class multivariate model (for initial values)...\n")
  
  tryCatch({
    # Build the formula for multlcmm
    # Format: outcome1 + outcome2 + outcome3 ~ time_months
    formula_str <- paste(paste(outcome_vars, collapse = " + "), "~ time_months")
    fixed_formula <- as.formula(formula_str)
    
    cat("  Formula:", formula_str, "\n")
    
    model_1class <- multlcmm(
      fixed = fixed_formula,
      random = ~ time_months,
      subject = "subjectkey_numeric",
      ng = 1,
      data = data_analysis,
      verbose = FALSE
    )
    
    cat("  1-class model converged successfully\n")
    cat("  LogLik:", model_1class$loglik, "\n")
    
    # Now fit multi-class models using the 1-class model as starting values
    for (n_class in n_classes_vec) {
      cat("\n========================================\n")
      cat("Fitting", n_class, "class multivariate model...\n")
      cat("Time started:", format(Sys.time(), "%H:%M:%S"), "\n")
      cat("========================================\n")
      
      model_name <- paste0("mlcga_", n_class, "class")
      
      tryCatch({
        # Fit the model using multlcmm with initial values from 1-class model
        model <- multlcmm(
          fixed = fixed_formula,
          mixture = ~ time_months,
          random = ~ time_months,
          subject = "subjectkey_numeric",
          ng = n_class,
          data = data_analysis,
          B = model_1class,  # Use 1-class model as initial values
          verbose = FALSE,
          maxiter = 500
        )
        
        models[[model_name]] <- model
        
        # Extract fit statistics
        fit_stats <- rbind(fit_stats, data.frame(
          n_classes = n_class,
          AIC = model$AIC,
          BIC = model$BIC,
          loglik = model$loglik,
          converged = model$conv,
          n_iter = model$niter,
          n_obs = nrow(data_analysis),
          n_subjects = length(unique(data_analysis$subjectkey)),
          n_outcomes = length(outcome_vars)
        ))
        
        cat("  AIC:", round(model$AIC, 2), "\n")
        cat("  BIC:", round(model$BIC, 2), "\n")
        cat("  Converged:", model$conv, "\n")
        cat("  N iterations:", model$niter, "\n")
        cat("  Time completed:", format(Sys.time(), "%H:%M:%S"), "\n")
        
      }, error = function(e) {
        cat("  ERROR:", e$message, "\n")
        fit_stats <<- rbind(fit_stats, data.frame(
          n_classes = n_class,
          AIC = NA,
          BIC = NA,
          loglik = NA,
          converged = FALSE,
          n_iter = NA,
          n_obs = nrow(data_analysis),
          n_subjects = length(unique(data_analysis$subjectkey)),
          n_outcomes = length(outcome_vars)
        ))
      })
    }
    
  }, error = function(e) {
    cat("  ERROR fitting 1-class model:", e$message, "\n")
    stop("Cannot proceed without successful 1-class model")
  })
  
  return(list(
    models = models, 
    fit_stats = fit_stats,
    outcome_vars = outcome_vars,
    data_analysis = data_analysis
  ))
}

# ============================================================================
# RUN ANALYSIS
# ============================================================================

cat("\n============================================\n")
cat("STARTING MULTIVARIATE LCGA ANALYSIS\n")
cat("============================================\n")

# Determine which variables to analyze (ALL variables together)
outcome_vars <- intersect(cbcl_vars, names(merged_data))

if (length(additional_data_list) > 0) {
  for (file_info in additional_files) {
    additional_vars <- intersect(file_info$vars, names(merged_data))
    outcome_vars <- c(outcome_vars, additional_vars)
  }
}

cat("\n*** IMPORTANT: Analyzing ALL outcomes TOGETHER in ONE multivariate model ***\n")
cat("Outcomes:", paste(outcome_vars, collapse = ", "), "\n")

results <- fit_mlcga_models(
  data = merged_data,
  outcome_vars = outcome_vars,
  n_classes_vec = n_classes_to_test,
  missingness_strategy = missingness_strategy
)

# ============================================================================
# MODEL COMPARISON
# ============================================================================

cat("\n============================================\n")
cat("MODEL FIT STATISTICS\n")
cat("============================================\n")

print(results$fit_stats)

# Save fit statistics
fit_stats_file <- file.path(output_dir, "mlcga_fit_statistics.csv")
write.csv(results$fit_stats, fit_stats_file, row.names = FALSE)
cat("\nFit statistics saved to:", fit_stats_file, "\n")

# Find best fitting model (by BIC)
best_model_row <- results$fit_stats %>%
  filter(converged == TRUE) %>%
  slice_min(BIC, n = 1)

if (nrow(best_model_row) == 0) {
  stop("No models converged successfully. Check your data and model specification.")
}

cat("\n\nBEST FITTING MODEL (by BIC):\n")
print(best_model_row)

# ============================================================================
# EXTRACT AND VISUALIZE RESULTS FOR BEST MODEL
# ============================================================================

n_class <- best_model_row$n_classes
model_name <- paste0("mlcga_", n_class, "class")
model <- results$models[[model_name]]

if (!is.null(model)) {
  cat("\n\n========================================\n")
  cat("Results for MULTIVARIATE model with", n_class, "classes\n")
  cat("========================================\n")
  
  # Print summary
  print(summary(model))
  
  # Get posterior class probabilities
  posteriors <- model$pprob
  
  # Map subjectkey_numeric back to subjectkey
  posteriors_with_id <- results$data_analysis %>%
    select(subjectkey, subjectkey_numeric) %>%
    distinct() %>%
    left_join(posteriors, by = c("subjectkey_numeric" = "subject"))
  
  # Save posterior probabilities
  posteriors_file <- file.path(output_dir, 
                                paste0("posteriors_mlcga_", n_class, "class.csv"))
  write.csv(posteriors_with_id, posteriors_file, row.names = FALSE)
  cat("\nPosterior probabilities saved to:", posteriors_file, "\n")
  
  # Add class assignment to data
  data_with_class <- results$data_analysis %>%
    left_join(posteriors_with_id[, c("subjectkey", "class")], by = "subjectkey")
  
  # Create plots for EACH outcome variable
  for (outcome in results$outcome_vars) {
    cat("\n--- Plotting trajectories for:", outcome, "---\n")
    
    # Plot individual trajectories by class
    p1 <- ggplot(data_with_class, 
                 aes(x = time_months, y = !!sym(outcome), 
                     group = subjectkey, color = as.factor(class))) +
      geom_line(alpha = 0.2) +
      stat_smooth(aes(group = class), method = "loess", se = TRUE, size = 1.5) +
      labs(title = paste0("Individual Trajectories: ", outcome, " (MLCGA ", n_class, " classes)"),
           x = "Time (months)",
           y = outcome,
           color = "Class") +
      theme_minimal() +
      theme(legend.position = "bottom",
            text = element_text(size = 12))
    
    print(p1)
    
    # Save plot
    plot_file <- file.path(output_dir, 
                           paste0("trajectories_", outcome, "_mlcga_", n_class, "class.png"))
    ggsave(plot_file, plot = p1, width = 10, height = 6, dpi = 300)
    cat("  Saved plot to:", plot_file, "\n")
    
    # Plot mean trajectories by class
    mean_trajectories <- data_with_class %>%
      group_by(class, time_months) %>%
      summarise(
        mean_value = mean(!!sym(outcome), na.rm = TRUE),
        se = sd(!!sym(outcome), na.rm = TRUE) / sqrt(n()),
        .groups = "drop"
      )
    
    p2 <- ggplot(mean_trajectories, 
                 aes(x = time_months, y = mean_value, color = as.factor(class))) +
      geom_line(size = 1.5) +
      geom_point(size = 3) +
      geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                    width = 2) +
      labs(title = paste0("Mean Trajectories: ", outcome, " (MLCGA ", n_class, " classes)"),
           x = "Time (months)",
           y = paste("Mean", outcome),
           color = "Class") +
      theme_minimal() +
      theme(legend.position = "bottom",
            text = element_text(size = 12))
    
    print(p2)
    
    # Save mean plot
    mean_plot_file <- file.path(output_dir, 
                                paste0("mean_trajectories_", outcome, "_mlcga_", n_class, "class.png"))
    ggsave(mean_plot_file, plot = p2, width = 10, height = 6, dpi = 300)
    cat("  Saved mean plot to:", mean_plot_file, "\n")
  }
  
  # Print class proportions and characteristics
  cat("\n========================================\n")
  cat("CLASS CHARACTERISTICS\n")
  cat("========================================\n")
  
  cat("\nClass proportions:\n")
  class_props <- table(posteriors_with_id$class) / nrow(posteriors_with_id)
  print(class_props)
  
  # Mean values by class, time, and outcome
  cat("\n\nMean values by class, time, and outcome:\n")
  for (outcome in results$outcome_vars) {
    cat("\n--- ", outcome, " ---\n")
    class_means <- data_with_class %>%
      group_by(class, eventname) %>%
      summarise(
        n = n(),
        mean = mean(!!sym(outcome), na.rm = TRUE),
        sd = sd(!!sym(outcome), na.rm = TRUE),
        .groups = "drop"
      ) %>%
      arrange(class, eventname)
    print(class_means)
  }
}

# ============================================================================
# SAVE FINAL DATASET WITH CLASS ASSIGNMENTS
# ============================================================================

cat("\n\n========================================\n")
cat("SAVING FINAL DATASET\n")
cat("========================================\n")

# Create final dataset with class assignments
final_data <- merged_data %>%
  left_join(
    posteriors_with_id %>% select(subjectkey, class) %>% rename(mlcga_class = class),
    by = "subjectkey"
  )

# Save final dataset
final_data_file <- file.path(output_dir, "data_with_mlcga_class_assignments.csv")
write.csv(final_data, final_data_file, row.names = FALSE)

cat("Final dataset saved to:", final_data_file, "\n")
cat("  Contains", nrow(final_data), "observations\n")
cat("  Contains", length(unique(final_data$subjectkey)), "unique subjects\n")

# Save model object for later use
model_file <- file.path(output_dir, paste0("mlcga_", n_class, "class_model.RData"))
save(model, file = model_file)
cat("\nModel object saved to:", model_file, "\n")

# ============================================================================
# FINALIZE LOGGING
# ============================================================================

cat("\n\n============================================\n")
cat("MULTIVARIATE LCGA ANALYSIS COMPLETE\n")
cat("============================================\n")
cat("Completed:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("All results saved to:", output_dir, "\n")
cat("  - Model fit statistics\n")
cat("  - Posterior probabilities (class assignments)\n")
cat("  - Trajectory plots for each outcome\n")
cat("  - Data with class assignments\n")
cat("  - Saved model object (.RData)\n")
cat("============================================\n")
cat("\nLog file saved to:", log_file, "\n")

# Close log connection and restore options
sink(type = "output")
close(log_con)
options(warn = original_warning)

message("\n*** Analysis complete. Check ", log_file, " for full output ***\n")