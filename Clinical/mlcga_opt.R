# Multivariate Latent Class Growth Analysis (MLCGA) Script for ABCD Data
# CORRECTED & OPTIMIZED VERSION with parallel processing and tqdm-style progress bars
# TRUE multivariate analysis - considers all variables simultaneously
# Handles ABCD-specific time variables and CBCL irritability data
# INCLUDES: Both LCGA-style plots (composite) AND multivariate plots (faceted)

# ============================================================================
# PACKAGE INSTALLATION AND LOADING
# ============================================================================

# Install required packages if not already installed
required_packages <- c("lcmm", "dplyr", "ggplot2", "tidyr", "readr", 
                       "data.table", "parallel", "doParallel", "foreach", 
                       "progress", "pryr", "RColorBrewer", "gridExtra")

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
library(RColorBrewer)
library(gridExtra)

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS
# ============================================================================

# Output directory
output_dir <- "/home/rif17002/honors_thesis/mlcga_results"

# Base directory for CBCL data
cbcl_dir <- "/home/rif17002/honors_thesis/ABCD_CBCL/Release_5.0"

# Additional data files (optional) - other variables you want to include
# Each element should be a list with 'file' path and 'vars' to extract
additional_files <- list(
  # Variables to look at:
  # ksads_3_226_p, ksads_15_433_p, ksads_15_432_p, ksads_15_91_p, ksads_1_3_p
  # Try with and without 227 228 229 DMDD.

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

# CBCL data format
# "long" = one row per subject per timepoint (default for most analyses)
# "wide" = one row per subject with separate columns for each timepoint
cbcl_format <- "long"  # or "wide"

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

# Performance settings
use_parallel <- TRUE  # Set to FALSE to disable parallel processing
n_cores <- 16  # Leave one core free
max_iterations_initial <- 100  # Start with fewer iterations
max_iterations_final <- 500  # Maximum if needed

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
cat("MLCGA ANALYSIS LOG (CORRECTED VERSION)\n")
cat("Started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================\n\n")
cat("R Version:\n")
print(R.version.string)
cat("\nLoaded Packages:\n")
print(sessionInfo())
cat("\n")
cat("Log file:", log_file, "\n")
cat("Parallel processing:", use_parallel, "\n")
if (use_parallel) {
  cat("Number of cores:", n_cores, "\n")
}
cat("Initial memory usage:", format(mem_used(), units = "MB"), "\n\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Convert ABCD eventname to numeric months (FIXED)
convert_eventname_to_months <- function(eventname) {
  # Use dplyr::case_when with proper syntax
  months <- dplyr::case_when(
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

# Load and process CBCL long format files (OPTIMIZED with fread)
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
  
  # Progress bar for file loading
  cat("\nLoading", length(timepoints), "CBCL long format files...\n")
  pb <- progress_bar$new(
    format = "  [:bar] :percent | :current/:total files | eta: :eta",
    total = length(timepoints),
    clear = FALSE,
    width = 60
  )
  
  for (tp in timepoints) {
    pb$tick()
    
    file_suffix <- timepoint_files[tp]
    if (is.na(file_suffix)) {
      warning("No file mapping for timepoint: ", tp)
      next
    }
    
    file_path <- file.path(cbcl_dir, 
                           paste0("abcd_cbcl_irr_index_release5.0_", 
                                  file_suffix, "_long.csv"))
    
    if (file.exists(file_path)) {
      cat("  Loading:", basename(file_path), "...")
      
      # Use fread for faster reading
      df <- tryCatch({
        fread(file_path, showProgress = FALSE)
      }, error = function(e) {
        cat("  fread failed, trying read_csv...\n")
        read_csv(file_path, show_col_types = FALSE)
      })
      
      # Check for required columns
      if ("subjectkey" %in% names(df) || "src_subject_id" %in% names(df)) {
        # Standardize ID column name
        if ("src_subject_id" %in% names(df)) {
          setnames(df, "src_subject_id", "subjectkey")
        }
        
        # Add time information
        df[, eventname := tp]
        df[, time_months := convert_eventname_to_months(tp)]
        
        # Select relevant columns
        id_col <- "subjectkey"
        available_vars <- intersect(cbcl_vars, names(df))
        
        if (length(available_vars) > 0) {
          cols_to_keep <- c(id_col, "eventname", "time_months", available_vars)
          df <- df[, ..cols_to_keep]
          
          all_data[[tp]] <- df
          cat(" ✓ (", nrow(df), "rows,", length(available_vars), "vars)\n")
        } else {
          cat(" ✗ (no CBCL variables found)\n")
          warning("  Expected vars: ", paste(cbcl_vars, collapse = ", "), 
                  "\n  Available vars: ", paste(head(names(df), 20), collapse = ", "))
        }
      }
    } else {
      cat("  ✗ File not found:", basename(file_path), "\n")
      warning("File not found: ", file_path)
    }
  }
  
  cat("\n")
  
  if (length(all_data) > 0) {
    # Use rbindlist for faster combining
    combined <- rbindlist(all_data, use.names = TRUE, fill = TRUE)
    cat("\nCombined CBCL data:", nrow(combined), "rows x", ncol(combined), "columns\n")
    cat("Unique subjects:", uniqueN(combined$subjectkey), "\n")
    return(combined)
  } else {
    stop("No CBCL data could be loaded")
  }
}

# Load and process CBCL wide format files (FIXED)
load_cbcl_wide <- function(cbcl_dir, timepoints, cbcl_vars) {
  # Map timepoints to file suffixes and months
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
  
  # Progress bar for file loading
  cat("\nLoading", length(timepoints), "CBCL wide format files...\n")
  pb <- progress_bar$new(
    format = "  [:bar] :percent | :current/:total files | eta: :eta",
    total = length(timepoints),
    clear = FALSE,
    width = 60
  )
  
  for (tp in timepoints) {
    pb$tick()
    
    file_suffix <- timepoint_files[tp]
    if (is.na(file_suffix)) {
      warning("No file mapping for timepoint: ", tp)
      next
    }
    
    file_path <- file.path(cbcl_dir, 
                           paste0("abcd_cbcl_irr_index_release5.0_", 
                                  file_suffix, "_wide.csv"))
    
    if (file.exists(file_path)) {
      cat("  Loading:", basename(file_path), "...")
      
      # Use fread for faster reading with error handling
      df <- tryCatch({
        fread(file_path, showProgress = FALSE)
      }, error = function(e) {
        cat("  fread failed, trying read_csv...\n")
        tryCatch({
          read_csv(file_path, show_col_types = FALSE)
        }, error = function(e2) {
          cat("  ERROR: Could not read file\n")
          return(NULL)
        })
      })
      
      if (is.null(df)) next
      
      # Check for required columns
      if ("subjectkey" %in% names(df) || "src_subject_id" %in% names(df)) {
        # Standardize ID column name
        if ("src_subject_id" %in% names(df)) {
          setnames(df, "src_subject_id", "subjectkey")
        }
        
        # Select relevant columns (ID + CBCL vars)
        id_col <- "subjectkey"
        available_vars <- intersect(cbcl_vars, names(df))
        
        if (length(available_vars) > 0) {
          cols_to_keep <- c(id_col, available_vars)
          df <- df[, ..cols_to_keep]
          
          # Convert to long format
          # Add suffixes to variable names to indicate timepoint
          time_suffix <- paste0("_t", timepoint_months[tp])
          for (var in available_vars) {
            setnames(df, var, paste0(var, time_suffix))
          }
          
          all_data[[tp]] <- df
          cat(" ✓ (", nrow(df), "subjects,", length(available_vars), "vars)\n")
        } else {
          cat(" ✗ (no CBCL variables found)\n")
          warning("  Expected vars: ", paste(cbcl_vars, collapse = ", "),
                  "\n  Available vars: ", paste(head(names(df), 20), collapse = ", "))
        }
      }
    } else {
      cat("  ✗ File not found:", basename(file_path), "\n")
      warning("File not found: ", file_path)
    }
  }
  
  cat("\n")
  
  if (length(all_data) > 0) {
    # Merge all timepoints by subjectkey (each subject appears once)
    cat("\nMerging wide format data across timepoints...\n")
    combined <- Reduce(function(x, y) merge(x, y, by = "subjectkey", all = TRUE), 
                       all_data)
    
    cat("Combined CBCL data (wide):", nrow(combined), "subjects x", ncol(combined), "columns\n")
    
    # NOW convert from wide to long format for analysis (FIXED)
    cat("Converting wide format to long format for analysis...\n")
    
    # Identify all variable-time combinations
    var_cols <- setdiff(names(combined), "subjectkey")
    
    # Reshape to long format
    combined_long <- melt(combined, 
                          id.vars = "subjectkey",
                          measure.vars = var_cols,
                          variable.name = "var_time",
                          value.name = "value")
    
    # Split var_time into variable and time (FIXED: proper data.table syntax)
    combined_long[, variable := sub("_t[0-9]+$", "", var_time)]
    combined_long[, time_months := as.numeric(sub(".*_t", "", var_time))]
    combined_long[, var_time := NULL]
    
    # Convert eventname from time_months (FIXED: use proper data.table assignment)
    combined_long[, eventname := fifelse(
      time_months == 0, "baseline_year_1_arm_1",
      fifelse(time_months == 12, "1_year_follow_up_y_arm_1",
      fifelse(time_months == 24, "2_year_follow_up_y_arm_1",
      fifelse(time_months == 36, "3_year_follow_up_y_arm_1",
      fifelse(time_months == 48, "4_year_follow_up_y_arm_1",
      NA_character_)))))]
    
    # Pivot wider to get one column per variable with error handling
    combined_long <- tryCatch({
      dcast(combined_long, 
            subjectkey + eventname + time_months ~ variable,
            value.var = "value")
    }, error = function(e) {
      stop("Error in dcast: ", e$message, 
           "\nCheck that variable names don't have unexpected format")
    })
    
    cat("Converted to long format:", nrow(combined_long), "rows x", ncol(combined_long), "columns\n")
    cat("Unique subjects:", uniqueN(combined_long$subjectkey), "\n")
    
    return(combined_long)
  } else {
    stop("No CBCL data could be loaded")
  }
}

# Load additional variable files (OPTIMIZED with better error messages)
load_additional_file <- function(file_info, timepoints) {
  file_path <- file_info$file
  vars_to_extract <- file_info$vars
  time_var <- file_info$time_var
  
  if (!file.exists(file_path)) {
    warning("File not found: ", file_path)
    return(NULL)
  }
  
  cat("\nLoading additional file:", file_path, "\n")
  
  # Try fread first (fastest), then read_tsv, then read_csv
  df <- tryCatch({
    fread(file_path, showProgress = FALSE)
  }, error = function(e) {
    cat("  fread failed, trying read_tsv...\n")
    tryCatch({
      read_tsv(file_path, show_col_types = FALSE, guess_max = 10000)
    }, error = function(e2) {
      cat("  TSV read failed, trying CSV...\n")
      tryCatch({
        read_csv(file_path, show_col_types = FALSE, guess_max = 10000)
      }, error = function(e3) {
        cat("  All read attempts failed\n")
        return(NULL)
      })
    })
  })
  
  if (is.null(df)) {
    warning("Could not read file: ", file_path)
    return(NULL)
  }
  
  # Convert to data.table if not already
  if (!is.data.table(df)) {
    setDT(df)
  }
  
  cat("  File loaded successfully:", nrow(df), "rows,", ncol(df), "columns\n")
  cat("  First few column names:", paste(names(df)[1:min(10, ncol(df))], collapse = ", "), "\n")
  
  # Identify ID column - check if subjectkey exists
  if ("subjectkey" %in% names(df)) {
    cat("  Found 'subjectkey' column\n")
  } else if ("src_subject_id" %in% names(df)) {
    cat("  Found 'src_subject_id' column, renaming to 'subjectkey'\n")
    setnames(df, "src_subject_id", "subjectkey")
  } else {
    # Print ALL column names to help debug
    cat("  ERROR: No subject ID column found!\n")
    cat("  All columns:", paste(names(df), collapse = ", "), "\n")
    warning("No subject ID column (subjectkey or src_subject_id) found in ", file_path)
    return(NULL)
  }
  
  # Check for time variable
  if (!time_var %in% names(df)) {
    cat("  ERROR: Time variable '", time_var, "' not found!\n")
    cat("  Available columns:", paste(names(df), collapse = ", "), "\n")
    warning("Time variable '", time_var, "' not found in ", file_path)
    return(NULL)
  }
  
  # Filter to relevant timepoints
  df <- df[get(time_var) %in% timepoints]
  
  if (nrow(df) == 0) {
    warning("No rows match the specified timepoints in ", file_path)
    return(NULL)
  }
  
  df[, eventname := get(time_var)]
  df[, time_months := convert_eventname_to_months(get(time_var))]
  
  # Select relevant variables
  available_vars <- intersect(vars_to_extract, names(df))
  
  if (length(available_vars) == 0) {
    cat("  ERROR: None of requested variables found!\n")
    cat("  Requested:", paste(vars_to_extract, collapse = ", "), "\n")
    cat("  Available:", paste(head(names(df), 20), collapse = ", "), "\n")
    warning("None of the requested variables found in ", file_path)
    return(NULL)
  }
  
  if (length(available_vars) < length(vars_to_extract)) {
    missing_vars <- setdiff(vars_to_extract, available_vars)
    cat("  WARNING: Some variables not found:", paste(missing_vars, collapse = ", "), "\n")
  }
  
  cols_to_keep <- c("subjectkey", "eventname", "time_months", available_vars)
  df <- df[, ..cols_to_keep]
  
  cat("  Loaded", nrow(df), "rows with variables:", paste(available_vars, collapse = ", "), "\n")
  
  return(df)
}

# ============================================================================
# TQDM-STYLE PROGRESS BAR FOR MODEL FITTING
# ============================================================================

create_model_progress_bar <- function(total_classes, stage = "fitting") {
  # Create a custom progress bar similar to tqdm
  pb <- progress_bar$new(
    format = paste0("  ", stage, " [:bar] :percent | :current/:total models | ",
                   "elapsed: :elapsed | eta: :eta | :model_info"),
    total = total_classes,
    clear = FALSE,
    width = 80,
    complete = "█",
    incomplete = "░"
  )
  return(pb)
}

# Enhanced model fitting with tqdm-style progress
fit_model_adaptive <- function(formula_str, data_analysis, ng, B_model = NULL, 
                              max_iter_sequence = c(100, 250, 500),
                              progress_callback = NULL) {
  
  fixed_formula <- as.formula(formula_str)
  
  for (iter_idx in seq_along(max_iter_sequence)) {
    max_iter <- max_iter_sequence[iter_idx]
    
    # Update progress if callback provided
    if (!is.null(progress_callback)) {
      progress_callback(paste0("iter:", max_iter))
    }
    
    tryCatch({
      if (is.null(B_model)) {
        # First model (1-class)
        model <- multlcmm(
          fixed = fixed_formula,
          random = ~ time_months,
          subject = "subjectkey_numeric",
          ng = ng,
          data = as.data.frame(data_analysis),
          verbose = FALSE,
          maxiter = max_iter
        )
      } else {
        # Multi-class models with initial values
        model <- multlcmm(
          fixed = fixed_formula,
          mixture = ~ time_months,
          random = ~ time_months,
          subject = "subjectkey_numeric",
          ng = ng,
          data = as.data.frame(data_analysis),
          B = B_model,
          verbose = FALSE,
          maxiter = max_iter
        )
      }
      
      if (model$conv == 1 || model$conv == 2) {
        if (!is.null(progress_callback)) {
          progress_callback(paste0("✓ conv:", model$conv))
        }
        return(model)
      } else {
        if (iter_idx < length(max_iter_sequence)) {
          if (!is.null(progress_callback)) {
            progress_callback(paste0("retry +", max_iter_sequence[iter_idx + 1] - max_iter, "iter"))
          }
        }
      }
      
    }, error = function(e) {
      if (!is.null(progress_callback)) {
        progress_callback(paste0("ERROR: ", substr(e$message, 1, 30)))
      }
      if (iter_idx == length(max_iter_sequence)) {
        return(NULL)
      }
    })
  }
  
  warning("Model did not converge even with maximum iterations")
  return(model)  # Return last attempt even if not converged
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

# Overall workflow progress
cat("\n")
cat("╔════════════════════════════════════════════════════════════════╗\n")
cat("║                   MLCGA ANALYSIS WORKFLOW                      ║\n")
cat("╠════════════════════════════════════════════════════════════════╣\n")
cat("║ Step 1: Load CBCL data                                    [ ] ║\n")
cat("║ Step 2: Load additional data files                       [ ] ║\n")
cat("║ Step 3: Merge datasets                                   [ ] ║\n")
cat("║ Step 4: Prepare data for analysis                        [ ] ║\n")
cat("║ Step 5: Fit MLCGA models                                 [ ] ║\n")
cat("║ Step 6: Select best model                                [ ] ║\n")
cat("║ Step 7: Generate visualizations                          [ ] ║\n")
cat("║ Step 8: Save results                                     [ ] ║\n")
cat("╚════════════════════════════════════════════════════════════════╝\n")
cat("\n")

# Load CBCL data
cat("\n[Step 1/8] Loading CBCL data (format:", cbcl_format, ")...\n")

if (cbcl_format == "wide") {
  cbcl_data <- load_cbcl_wide(cbcl_dir, timepoints_to_include, cbcl_vars)
} else if (cbcl_format == "long") {
  cbcl_data <- load_cbcl_long(cbcl_dir, timepoints_to_include, cbcl_vars)
} else {
  stop("Invalid cbcl_format. Must be 'wide' or 'long'")
}

cat("✓ Step 1 complete: CBCL data loaded\n")

# Load additional files if specified
additional_data_list <- list()
if (length(additional_files) > 0) {
  cat("\n[Step 2/8] Loading", length(additional_files), "additional data file(s)...\n")
  
  pb_add <- progress_bar$new(
    format = "  [:bar] :percent | :current/:total files | eta: :eta",
    total = length(additional_files),
    clear = FALSE,
    width = 60
  )
  
  for (i in seq_along(additional_files)) {
    pb_add$tick()
    add_data <- load_additional_file(additional_files[[i]], timepoints_to_include)
    if (!is.null(add_data)) {
      additional_data_list[[i]] <- add_data
    }
  }
  cat("\n")
  cat("✓ Step 2 complete: Additional data loaded\n")
} else {
  cat("\n[Step 2/8] No additional files to load\n")
  cat("✓ Step 2 complete: Skipped (no additional files)\n")
}

# Merge all data sources (OPTIMIZED with data.table merge)
if (length(additional_data_list) > 0) {
  cat("\n[Step 3/8] Merging", length(additional_data_list) + 1, "data sources...\n")
  
  pb_merge <- progress_bar$new(
    format = "  [:bar] :percent | :current/:total merges | eta: :eta",
    total = length(additional_data_list),
    clear = FALSE,
    width = 60
  )
  
  merged_data <- cbcl_data
  
  for (add_data in additional_data_list) {
    pb_merge$tick()
    
    # Ensure both are data.tables
    if (!is.data.table(merged_data)) setDT(merged_data)
    if (!is.data.table(add_data)) setDT(add_data)
    
    # Use data.table merge (faster than dplyr join)
    merged_data <- merge(merged_data, add_data, 
                         by = c("subjectkey", "eventname", "time_months"),
                         all = TRUE)
  }
  
  cat("\nMerged data dimensions:", nrow(merged_data), "rows x", ncol(merged_data), "columns\n")
  cat("✓ Step 3 complete: Data merged\n")
} else {
  cat("\n[Step 3/8] No merging needed (single data source)\n")
  merged_data <- cbcl_data
  cat("✓ Step 3 complete: Skipped (single source)\n")
}

# Remove rows with missing time
merged_data <- merged_data[!is.na(time_months)]

# CRITICAL FIX: Create stable ID mapping
cat("\nCreating stable subject ID mapping...\n")
id_mapping <- unique(merged_data[, .(subjectkey)])
id_mapping[, subjectkey_numeric := .I]

# Add numeric IDs to merged data
merged_data <- merge(merged_data, id_mapping, by = "subjectkey", all.x = TRUE)

cat("Converted", nrow(id_mapping), "subject IDs to numeric format\n")
cat("Memory after data loading:", format(mem_used(), units = "MB"), "\n")
cat("\n✓ Step 4 complete: Data prepared for analysis\n")

# Summary statistics
cat("\n============================================\n")
cat("DATA SUMMARY\n")
cat("============================================\n")
cat("Total observations:", nrow(merged_data), "\n")
cat("Unique subjects:", uniqueN(merged_data$subjectkey), "\n")
cat("Time points:\n")
print(table(merged_data$eventname))
cat("\nVariables in dataset:\n")
print(names(merged_data))

# Check missingness (OPTIMIZED)
cat("\nMissing data summary:\n")
missing_summary <- merged_data[, lapply(.SD, function(x) sum(is.na(x)))] %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  mutate(pct_missing = round(n_missing / nrow(merged_data) * 100, 2)) %>%
  arrange(desc(n_missing))

print(missing_summary)

# ============================================================================
# MODEL FITTING FUNCTIONS (OPTIMIZED WITH TQDM-STYLE PROGRESS)
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
  
  # Handle missingness according to strategy
  if (missingness_strategy == "complete_case") {
    # Filter data to complete cases for all outcomes (OPTIMIZED)
    data_analysis <- data[complete.cases(data[, ..outcome_vars])]
    
    cat("\nUsing COMPLETE CASE analysis\n")
    cat("N observations with complete data across all outcomes:", nrow(data_analysis), "\n")
  } else {
    # Use all available data (pairwise deletion)
    data_analysis <- data
    cat("\nUsing PAIRWISE deletion (all available data)\n")
    cat("N observations:", nrow(data_analysis), "\n")
  }
  
  cat("N subjects:", uniqueN(data_analysis$subjectkey), "\n")
  
  if (nrow(data_analysis) < 100) {
    stop("Too few observations with data. Consider changing missingness strategy or checking data.")
  }
  
  # Validation: Check sample size per class (rough estimate)
  min_sample_per_class <- nrow(data_analysis) / max(n_classes_vec) / 10
  if (min_sample_per_class < 20) {
    warning("Small sample size may lead to unstable estimates. ",
            "Estimated min subjects per class: ", round(min_sample_per_class), 
            ". Consider reducing number of classes.")
  }
  
  # Check if data is in correct format for multlcmm
  subjects_per_row <- data_analysis[, .(n_rows = .N), by = subjectkey]
  
  if (any(subjects_per_row$n_rows > 1)) {
    cat("\n*** WARNING: Data is in LONG format ***\n")
    cat("*** multlcmm typically expects WIDE format for multivariate analysis ***\n")
    cat("*** Proceeding with long format - verify results carefully ***\n")
    cat("*** Number of observations per subject (first 10):\n")
    print(head(subjects_per_row, 10))
  }
  
  # First, fit a 1-class model to get initial values
  cat("\n========================================\n")
  cat("Fitting 1-class multivariate model\n")
  cat("(for initial values)\n")
  cat("========================================\n")
  cat("Time started:", format(Sys.time(), "%H:%M:%S"), "\n")
  
  # Build the formula for multlcmm
  # Format: outcome1 + outcome2 + outcome3 ~ time_months
  formula_str <- paste(paste(outcome_vars, collapse = " + "), "~ time_months")
  cat("  Formula:", formula_str, "\n\n")
  
  # Cache file for 1-class model
  cache_file <- file.path(output_dir, "cache_model_1class.RData")
  
  if (file.exists(cache_file)) {
    cat("  Loading cached 1-class model...\n")
    load(cache_file)
    cat("  ✓ Loaded from cache\n")
  } else {
    cat("  Fitting 1-class model...\n")
    
    # Create progress indicator for 1-class model
    pb_1class <- progress_bar$new(
      format = "  Initializing [:bar] :model_info",
      total = 1,
      clear = FALSE,
      width = 70,
      complete = "█",
      incomplete = "░"
    )
    
    model_1class <- fit_model_adaptive(
      formula_str = formula_str,
      data_analysis = data_analysis,
      ng = 1,
      B_model = NULL,
      max_iter_sequence = c(max_iterations_initial, max_iterations_final),
      progress_callback = function(info) {
        pb_1class$tick(tokens = list(model_info = info))
      }
    )
    
    pb_1class$terminate()
    
    if (is.null(model_1class)) {
      stop("Failed to fit 1-class model")
    }
    
    # Save for future use
    save(model_1class, file = cache_file)
    cat("  ✓ Cached 1-class model saved\n")
  }
  
  cat("  LogLik:", round(model_1class$loglik, 2), "\n")
  cat("  Time completed:", format(Sys.time(), "%H:%M:%S"), "\n")
  
  # Now fit multi-class models
  if (use_parallel && length(n_classes_vec) > 1) {
    cat("\n========================================\n")
    cat("PARALLEL MODEL FITTING\n")
    cat("========================================\n")
    cat("Using", n_cores, "cores\n")
    cat("Fitting", length(n_classes_vec), "models in parallel...\n\n")
    
    # Setup parallel backend
    cl <- makeCluster(n_cores)
    registerDoParallel(cl)
    
    # Export necessary objects to cluster (FIXED: reduced memory footprint)
    clusterExport(cl, c("formula_str", "data_analysis", "max_iterations_initial", 
                        "max_iterations_final"),
                  envir = environment())
    
    # Export model B values (parameters only, not full object)
    B_values <- model_1class$best
    clusterExport(cl, "B_values", envir = environment())
    
    # Create progress bar for parallel execution
    pb_parallel <- create_model_progress_bar(length(n_classes_vec), "Parallel")
    
    # Fit models in parallel
    results_list <- foreach(n_class = n_classes_vec, 
                            .packages = c("lcmm", "dplyr"),
                            .errorhandling = "pass",
                            .combine = 'list',
                            .multicombine = TRUE) %dopar% {
      
      # Reconstruct B_model from parameters
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
          maxiter = max_iterations_final
        )
      }, error = function(e) {
        NULL
      })
      
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
          model = NULL,
          n_classes = n_class,
          AIC = NA,
          BIC = NA,
          loglik = NA,
          converged = FALSE,
          n_iter = NA
        )
      }
    }
    
    # Stop cluster
    stopCluster(cl)
    
    # Process results with progress updates
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
      
      # Update progress bar
      status <- ifelse(result$converged, "✓", "✗")
      pb_parallel$tick(tokens = list(
        model_info = paste0(n_class, "-class ", status, " BIC:", round(result$BIC, 0))
      ))
    }
    
    pb_parallel$terminate()
    
  } else {
    # Sequential fitting with enhanced tqdm-style progress bar
    cat("\n========================================\n")
    cat("SEQUENTIAL MODEL FITTING\n")
    cat("========================================\n\n")
    
    pb <- create_model_progress_bar(length(n_classes_vec), "Sequential")
    
    for (idx in seq_along(n_classes_vec)) {
      n_class <- n_classes_vec[idx]
      model_name <- paste0("mlcga_", n_class, "class")
      
      model <- fit_model_adaptive(
        formula_str = formula_str,
        data_analysis = data_analysis,
        ng = n_class,
        B_model = model_1class,
        max_iter_sequence = c(max_iterations_initial, max_iterations_final),
        progress_callback = function(info) {
          pb$tick(tokens = list(
            model_info = paste0(n_class, "-class ", info)
          ))
        }
      )
      
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
        
        # Final tick with results
        pb$tick(tokens = list(
          model_info = paste0(n_class, "-class ✓ AIC:", round(model$AIC, 0), 
                            " BIC:", round(model$BIC, 0))
        ))
      } else {
        fit_stats <- rbind(fit_stats, data.frame(
          n_classes = n_class,
          AIC = NA,
          BIC = NA,
          loglik = NA,
          converged = FALSE,
          n_iter = NA,
          n_obs = nrow(data_analysis),
          n_subjects = uniqueN(data_analysis$subjectkey),
          n_outcomes = length(outcome_vars)
        ))
        
        pb$tick(tokens = list(
          model_info = paste0(n_class, "-class ✗ FAILED")
        ))
      }
      
      # Garbage collection after each model
      gc(verbose = FALSE)
    }
    
    pb$terminate()
  }
  
  cat("\n")
  cat("Memory after model fitting:", format(mem_used(), units = "MB"), "\n")
  
  return(list(
    models = models, 
    fit_stats = fit_stats,
    outcome_vars = outcome_vars,
    data_analysis = data_analysis,
    id_mapping = id_mapping
  ))
}

# ============================================================================
# RUN ANALYSIS
# ============================================================================

cat("\n============================================\n")
cat("STARTING MULTIVARIATE LCGA ANALYSIS\n")
cat("============================================\n")

cat("\n[Step 5/8] Fitting MLCGA models...\n")

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
  missingness_strategy = missingness_strategy,
  use_parallel = use_parallel,
  n_cores = n_cores
)

cat("\n✓ Step 5 complete: All models fitted\n")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

cat("\n============================================\n")
cat("MODEL FIT STATISTICS\n")
cat("============================================\n")

cat("\n[Step 6/8] Selecting best model...\n")

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

# FIXED: Extract n_class BEFORE using it
n_class <- best_model_row$n_classes

cat("\n\nBEST FITTING MODEL (by BIC):\n")
print(best_model_row)

cat("\n✓ Step 6 complete: Best model selected (", n_class, "classes)\n")

# ============================================================================
# EXTRACT AND VISUALIZE RESULTS FOR BEST MODEL
# ============================================================================

cat("\n[Step 7/8] Generating visualizations...\n")

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
  
  # Check what the ID column is actually named and convert to proper type
  cat("Posterior probability columns:", paste(names(posteriors), collapse = ", "), "\n")
  posteriors_df <- as.data.frame(posteriors)
  id_col_name <- names(posteriors_df)[1]
  posteriors_df[[id_col_name]] <- as.integer(posteriors_df[[id_col_name]])
  
  # FIXED: Use the stable ID mapping with proper type conversion
  posteriors_with_id <- results$id_mapping %>%
    left_join(posteriors_df, by = setNames(id_col_name, "subjectkey_numeric"))
  
  # Identify the class column
  if ("class" %in% colnames(posteriors_with_id)) {
    assigned_class <- posteriors_with_id$class
  } else {
    prob_cols <- grep("^prob", colnames(posteriors_with_id), value = TRUE)
    assigned_class <- apply(as.matrix(posteriors_with_id[, prob_cols, drop = FALSE]), 1, which.max)
  }
  
  # Save posterior probabilities
  posteriors_file <- file.path(output_dir, 
                                paste0("posteriors_mlcga_", n_class, "class.csv"))
  write.csv(posteriors_with_id, posteriors_file, row.names = FALSE)
  cat("\nPosterior probabilities saved to:", posteriors_file, "\n")
  
  # Add class assignment to data (OPTIMIZED with data.table)
  data_with_class <- merge(
    results$data_analysis,
    data.frame(
      subjectkey = posteriors_with_id$subjectkey,
      latent_class = factor(assigned_class)
    ),
    by = "subjectkey",
    all.x = TRUE
  )
  
  setDT(data_with_class)
  
  # OPTIMIZED PLOTTING: Create all plot data at once, then facet
  cat("\n========================================\n")
  cat("CREATING PLOTS\n")
  cat("========================================\n")
  
  # Prepare data for plotting (all outcomes at once)
  plot_data <- data_with_class %>%
    pivot_longer(
      cols = all_of(results$outcome_vars),
      names_to = "outcome",
      values_to = "value"
    ) %>%
    filter(!is.na(value))
  
  # Convert to data.table for faster operations
  setDT(plot_data)
  
  # ========================================================================
  # PART 1: LCGA-STYLE PLOTS (composite score on single graph)
  # ========================================================================
  
  cat("\n--- Creating LCGA-style plots (composite score) ---\n")
  
  # Create composite score (average across all outcomes)
  data_with_class[, composite_score := rowMeans(.SD, na.rm = TRUE), 
                  .SDcols = results$outcome_vars]
  
  # Calculate class sizes
  class_sizes <- data_with_class[, .N, by = .(latent_class, subjectkey)][
    , .N, by = latent_class
  ]
  setnames(class_sizes, "N", "n_subjects")
  
  # Calculate trajectory means for composite
  traj_means <- data_with_class[, .(
    mean_outcome = mean(composite_score, na.rm = TRUE),
    se_outcome = sd(composite_score, na.rm = TRUE) / sqrt(.N),
    n = .N
  ), by = .(latent_class, time_months)]
  
  traj_means <- merge(traj_means, class_sizes, by = "latent_class")
  traj_means[, class_label := sprintf("Class %s (n=%d)", latent_class, n_subjects)]
  
  # Define TIMEFRAMES for x-axis
  TIMEFRAMES <- c(0, 12, 24, 36, 48)

  # Fixed color palette (matching LCGA)
  base_palette <- c(
    "1" = "#4f8bc8",   # Class 1 - blue
    "2" = "#8b3fca",   # Class 2 - purple
    "3" = "#cb6587",   # Class 3 - pink
    "4" = "#f7a258",   # Class 4 - orange
    "5" = "#7dc383"    # Class 5 - green
  )

  # Get actual class numbers present in the data
  actual_classes <- as.character(sort(unique(assigned_class)))
  cat("Actual classes found in data:", paste(actual_classes, collapse = ", "), "\n")

  # Create palette matching ONLY the classes that exist
  class_palette <- base_palette[actual_classes]
  
  class_label_lookup <- unique(traj_means[, .(latent_class, class_label)])
  setorder(class_label_lookup, latent_class)
  
  class_label_palette <- setNames(
    class_palette[as.character(class_label_lookup$latent_class)],
    class_label_lookup$class_label
  )
  
  # ── BIC Plot ──
  cat("  Creating BIC plot...\n")
  p_bic <- ggplot(results$fit_stats, aes(x = n_classes, y = BIC)) +
    geom_line(color = "#2C7BB6", linewidth = 1) +
    geom_point(color = "#2C7BB6", size = 3) +
    geom_vline(xintercept = n_class, linetype = "dashed", color = "red", linewidth = 0.8) +
    annotate("text", x = n_class + 0.15, y = max(results$fit_stats$BIC, na.rm = TRUE),
             label = sprintf("Optimal\nk = %d", n_class), color = "red", size = 3.5, hjust = 0) +
    labs(title = "Model Selection: BIC by Number of Classes",
         x = "Number of Latent Classes", y = "BIC (lower = better)") +
    theme_bw(base_size = 12)
  
  ggsave(file.path(output_dir, "mlcga_bic_plot.png"),
         p_bic, width = 7, height = 5, dpi = 150)
  
  # ── Mean trajectory plot (LCGA-style) ──
  cat("  Creating mean trajectories plot (composite)...\n")
  p_traj_mean <- ggplot(traj_means,
                        aes(x = time_months, y = mean_outcome,
                            color = class_label, group = class_label)) +
    geom_ribbon(aes(ymin = mean_outcome - se_outcome,
                    ymax = mean_outcome + se_outcome,
                    fill = class_label), alpha = 0.15, color = NA) +
    geom_line(linewidth = 1.4) +
    geom_point(size = 2.8) +
    scale_x_continuous(breaks = TIMEFRAMES, labels = paste0(TIMEFRAMES, "m")) +
    labs(title    = sprintf("MLCGA Mean Trajectories (%d-Class Solution)", n_class),
         subtitle = "Mean +/- SE of Composite Score (averaged across all outcomes)",
         x = "Time (months)", y = "Composite Score",
         color = "Latent Class", fill = "Latent Class") +
    scale_color_manual(values = class_label_palette) +
    scale_fill_manual(values  = class_label_palette) +
    theme_bw(base_size = 12) +
    theme(legend.position = "right")
  
  ggsave(file.path(output_dir, "mlcga_mean_trajectories.png"),
         p_traj_mean, width = 9, height = 6, dpi = 150)
  
  # ── Individual trajectories — ALL classes on one plot ──
  cat("  Creating individual trajectories plot (all classes)...\n")
  MAX_INDIV_PER_CLASS <- 300
  set.seed(2024)
  
  indiv_sample <- data_with_class[, .SD[sample(.N, min(.N, MAX_INDIV_PER_CLASS))], 
                                   by = latent_class]
  
  p_indiv_all <- ggplot(indiv_sample,
                      aes(x = time_months, y = composite_score,
                          group = interaction(subjectkey, latent_class),
                          color = as.character(latent_class))) +
    geom_line(alpha = 0.04, linewidth = 0.35) +
    geom_line(data = traj_means,
              aes(x = time_months, y = mean_outcome,
                  group = class_label, color = as.character(latent_class)),
              linewidth = 1.6, inherit.aes = FALSE) +
    scale_color_manual(values = class_palette,
                      name = "Latent Class",
                      labels = actual_classes) +
    scale_x_continuous(breaks = TIMEFRAMES, labels = paste0(TIMEFRAMES, "m")) +
    labs(title    = sprintf("Individual Trajectories — All Classes (%d-Class Solution)", n_class),
        subtitle = sprintf("Faint lines = individuals (alpha=0.04); bold = class mean | up to %d per class", MAX_INDIV_PER_CLASS),
        x = "Time (months)", y = "Composite Score") +
    theme_bw(base_size = 12) +
    theme(legend.position = "right")
  
  ggsave(file.path(output_dir, "mlcga_individual_all_classes.png"),
         p_indiv_all, width = 11, height = 7, dpi = 150)
  
  # ── One plot per class ──
  cat("  Creating individual plots for each class...\n")
  indiv_plots <- lapply(sort(unique(assigned_class)), function(cls) {
    cls_n      <- class_sizes$n_subjects[class_sizes$latent_class == cls]
    cls_colour <- class_palette[as.character(cls)]
    cls_mean   <- traj_means[latent_class == cls]
    cls_data   <- indiv_sample[latent_class == cls]
    
    ggplot(cls_data,
           aes(x = time_months, y = composite_score,
               group = subjectkey)) +
      geom_line(alpha = 0.06, linewidth = 0.35, color = cls_colour) +
      geom_line(data = cls_mean,
                aes(x = time_months, y = mean_outcome, group = 1),
                color = cls_colour, linewidth = 2.0, inherit.aes = FALSE) +
      scale_x_continuous(breaks = TIMEFRAMES, labels = paste0(TIMEFRAMES, "m")) +
      coord_cartesian(ylim = c(0, max(data_with_class$composite_score, na.rm = TRUE))) +
      labs(title    = sprintf("Class %s — Individual Trajectories", cls),
           subtitle = sprintf("n = %d subjects  |  bold line = class mean", cls_n),
           x = "Time (months)", y = "Composite Score") +
      theme_bw(base_size = 13)
  })
  
  for (i in seq_along(indiv_plots)) {
    cls   <- sort(unique(assigned_class))[i]
    fname <- sprintf("mlcga_class%s_individual_trajectories.png", cls)
    ggsave(file.path(output_dir, fname), indiv_plots[[i]], width = 8, height = 6, dpi = 150)
  }
  
  # ── Class size bar chart ──
  cat("  Creating class sizes bar chart...\n")
  p_size <- ggplot(class_sizes, aes(x = latent_class, y = n_subjects, fill = latent_class)) +
    geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
    scale_fill_manual(values = class_palette) +
    geom_text(aes(label = sprintf("%d\n(%.1f%%)", n_subjects,
                                  100 * n_subjects / sum(n_subjects))),
              vjust = -0.3, size = 3.5) +
    labs(title = "Class Membership Sizes",
         x = "Latent Class", y = "Number of Subjects") +
    theme_bw(base_size = 12)
  
  ggsave(file.path(output_dir, "mlcga_class_sizes.png"),
         p_size, width = 6, height = 5, dpi = 150)
  
  cat("  ✓ LCGA-style plots complete\n\n")
  
  # ========================================================================
  # PART 2: MULTIVARIATE PLOTS (separate panels for each outcome)
  # ========================================================================
  
  cat("--- Creating multivariate plots (faceted by outcome) ---\n")
  
  # Individual trajectories plot (faceted by outcome)
  cat("  Creating faceted individual trajectories...\n")
  p1 <- ggplot(plot_data, 
             aes(x = time_months, y = value, 
                 group = subjectkey, color = as.character(latent_class))) +
    geom_line(alpha = 0.04) +
    facet_wrap(~ outcome, scales = "free_y", ncol = 2) +
    labs(title = paste0("Individual Trajectories by Outcome (MLCGA ", n_class, " classes)"),
        x = "Time (months)",
        y = "Value",
        color = "Class") +
    scale_color_manual(values = class_palette,
                      labels = actual_classes) +
    theme_minimal() +
    theme(legend.position = "bottom",
          text = element_text(size = 10),
        strip.text = element_text(size = 11, face = "bold"))
  
  ggsave(file.path(output_dir, 
                   paste0("trajectories_all_outcomes_mlcga_", n_class, "class.png")),
         plot = p1, width = 12, height = 8, dpi = 300)
  
  # Mean trajectories by class (faceted by outcome)
  cat("  Creating faceted mean trajectories...\n")
  mean_trajectories_by_outcome <- plot_data[, .(
    mean_value = mean(value, na.rm = TRUE),
    se = sd(value, na.rm = TRUE) / sqrt(.N)
  ), by = .(latent_class, time_months, outcome)]
  
  p2 <- ggplot(mean_trajectories_by_outcome, 
             aes(x = time_months, y = mean_value, color = as.character(latent_class))) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    geom_errorbar(aes(ymin = mean_value - se, ymax = mean_value + se), 
                  width = 2, alpha = 0.6) +
    facet_wrap(~ outcome, scales = "free_y", ncol = 2) +
    labs(title = paste0("Mean Trajectories by Outcome (MLCGA ", n_class, " classes)"),
        x = "Time (months)",
        y = "Mean Value",
        color = "Class") +
    scale_color_manual(values = class_palette,
                      labels = actual_classes) +
    theme_minimal() +
    theme(legend.position = "bottom",
          text = element_text(size = 10),
        strip.text = element_text(size = 11, face = "bold"))
  
  ggsave(file.path(output_dir, 
                   paste0("mean_trajectories_all_outcomes_mlcga_", n_class, "class.png")),
         plot = p2, width = 12, height = 8, dpi = 300)
  
  # Create individual plots for each outcome
  cat("  Creating individual outcome plots...\n")
  
  pb_plots <- progress_bar$new(
    format = "  [:bar] :percent | :current/:total plots | eta: :eta",
    total = length(results$outcome_vars),
    clear = FALSE,
    width = 60
  )
  
  for (outcome in results$outcome_vars) {
    pb_plots$tick()
    
    outcome_data <- plot_data[outcome == !!outcome]
    
    p_ind <- ggplot(outcome_data, 
                aes(x = time_months, y = value, 
                    group = subjectkey, color = as.character(latent_class))) +
      geom_line(alpha = 0.2) +
      stat_smooth(aes(group = latent_class), method = "loess", se = TRUE, linewidth = 1.5) +
      scale_color_manual(values = class_palette,
                        labels = actual_classes) +
      labs(title = paste0("Individual Trajectories: ", outcome, " (MLCGA ", n_class, " classes)"),
          x = "Time (months)",
          y = outcome,
          color = "Class") +
      theme_minimal() +
      theme(legend.position = "bottom",
        text = element_text(size = 12))
    
    plot_file_ind <- file.path(output_dir, 
                               paste0("trajectories_", outcome, "_mlcga_", n_class, "class.png"))
    ggsave(plot_file_ind, plot = p_ind, width = 10, height = 6, dpi = 300)
  }
  
  pb_plots$terminate()
  
  cat("  ✓ Multivariate plots complete\n")
  cat("\n✓ Step 7 complete: All visualizations generated\n")
  
  # Print class proportions and characteristics
  cat("\n========================================\n")
  cat("CLASS CHARACTERISTICS\n")
  cat("========================================\n")
  
  cat("\nClass proportions:\n")
  class_props <- table(assigned_class) / length(assigned_class)
  print(class_props)
  
  # Mean values by class, time, and outcome (OPTIMIZED)
  cat("\n\nMean values by class, time, and outcome:\n")
  for (outcome in results$outcome_vars) {
    cat("\n--- ", outcome, " ---\n")
    class_means <- data_with_class[, .(
      n = .N,
      mean = mean(get(outcome), na.rm = TRUE),
      sd = sd(get(outcome), na.rm = TRUE)
    ), by = .(latent_class, eventname)][order(latent_class, eventname)]
    
    print(class_means)
  }
}

# ============================================================================
# SAVE FINAL DATASET WITH CLASS ASSIGNMENTS
# ============================================================================

cat("\n\n========================================\n")
cat("SAVING FINAL DATASET\n")
cat("========================================\n")

cat("\n[Step 8/8] Saving all results...\n")

# Create final dataset with class assignments (OPTIMIZED with data.table)
final_data <- merge(
  merged_data,
  data.frame(subjectkey = posteriors_with_id$subjectkey, 
             class = assigned_class),
  by = "subjectkey",
  all.x = TRUE
)

setDT(final_data)

# Rename class column
setnames(final_data, "class", "mlcga_class")

# Save final dataset
final_data_file <- file.path(output_dir, "data_with_mlcga_class_assignments.csv")
fwrite(final_data, final_data_file)

cat("Final dataset saved to:", final_data_file, "\n")
cat("  Contains", nrow(final_data), "observations\n")
cat("  Contains", uniqueN(final_data$subjectkey), "unique subjects\n")

# Save model object for later use
model_file <- file.path(output_dir, paste0("mlcga_", n_class, "class_model.RData"))
save(model, file = model_file)
cat("\nModel object saved to:", model_file, "\n")

# Save all results for reproducibility
all_results_file <- file.path(output_dir, "mlcga_all_results.RData")
save(results, file = all_results_file)
cat("All results object saved to:", all_results_file, "\n")

cat("\n✓ Step 8 complete: All results saved\n")

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

cat("\n\n========================================\n")
cat("PERFORMANCE SUMMARY\n")
cat("========================================\n")
cat("Final memory usage:", format(mem_used(), units = "MB"), "\n")
cat("Peak memory may have been higher during model fitting\n")

# ============================================================================
# FINALIZE LOGGING
# ============================================================================

cat("\n\n============================================\n")
cat("MULTIVARIATE LCGA ANALYSIS COMPLETE\n")
cat("============================================\n")

cat("\n")
cat("╔════════════════════════════════════════════════════════════════╗\n")
cat("║                   ALL STEPS COMPLETED ✓✓✓                     ║\n")
cat("╠════════════════════════════════════════════════════════════════╣\n")
cat("║ Step 1: Load CBCL data                                    [✓] ║\n")
cat("║ Step 2: Load additional data files                       [✓] ║\n")
cat("║ Step 3: Merge datasets                                   [✓] ║\n")
cat("║ Step 4: Prepare data for analysis                        [✓] ║\n")
cat("║ Step 5: Fit MLCGA models                                 [✓] ║\n")
cat("║ Step 6: Select best model                                [✓] ║\n")
cat("║ Step 7: Generate visualizations                          [✓] ║\n")
cat("║ Step 8: Save results                                     [✓] ║\n")
cat("╚════════════════════════════════════════════════════════════════╝\n")
cat("\n")

cat("Completed:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("All results saved to:", output_dir, "\n")
cat("  - LCGA-style plots (composite score):\n")
cat("    * mlcga_bic_plot.png\n")
cat("    * mlcga_mean_trajectories.png\n")
cat("    * mlcga_individual_all_classes.png\n")
cat("    * mlcga_class[X]_individual_trajectories.png\n")
cat("    * mlcga_class_sizes.png\n")
cat("  - Multivariate plots (faceted by outcome):\n")
cat("    * trajectories_all_outcomes_mlcga_[X]class.png\n")
cat("    * mean_trajectories_all_outcomes_mlcga_[X]class.png\n")
cat("    * trajectories_[outcome]_mlcga_[X]class.png (per outcome)\n")
cat("  - Model fit statistics\n")
cat("  - Posterior probabilities (class assignments)\n")
cat("  - Data with class assignments\n")
cat("  - Saved model object (.RData)\n")
cat("  - All results object for reproducibility\n")
cat("============================================\n")
cat("\nLog file saved to:", log_file, "\n")

# Close log connection and restore options
sink(type = "output")
close(log_con)
options(warn = original_warning)

message("\n*** Analysis complete. Check ", log_file, " for full output ***\n")
message("*** Final memory usage: ", format(mem_used(), units = "MB"), " ***\n")