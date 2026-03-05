# ==============================================================================
# Latent Class Growth Analysis (LCGA) on ABCD Irritability Data
# Dataset: abcd_cbcl_irr_index_release5.0_{TIMEFRAME}m_long.csv
# Timeframes: 0, 12, 24, 36, 48 months
# ==============================================================================

# --- 1. Install & Load Required Packages --------------------------------------

required_packages <- c("tidyverse", "lcmm", "ggplot2", "gridExtra", "knitr")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}
invisible(lapply(required_packages, install_if_missing))

library(tidyverse)
library(lcmm)      # Core package for LCGA via hlme()
library(ggplot2)
library(gridExtra)

# --- 2. Configuration ---------------------------------------------------------

BASE_PATH    <- "/shared/healthinfolab/datasets/ABCD/Irritability/Clinical_Data/Irritability/Release_5.0"
OUTPUT_DIR   <- Sys.getenv("LCGA_OUTPUT_DIR", unset = file.path(getwd(), "lcga_output"))
TIMEFRAMES   <- c(0, 12, 24, 36, 48)
OUTCOME_VAR  <- "cbcl_irr_index_cnst"   # Primary irritability index
ID_VAR       <- "src_subject_id"
TIME_VAR     <- "time_months"
MAX_CLASSES  <- 4                        # Maximum number of latent classes to test
SET_SEED     <- 2024

# --- 3. Load & Combine Data ---------------------------------------------------

load_data <- function(timeframes, base_path) {
  all_data <- map_dfr(timeframes, function(t) {
    fname <- sprintf("abcd_cbcl_irr_index_release5.0_%dm_long.csv", t)
    fpath <- file.path(base_path, fname)
    
    if (!file.exists(fpath)) {
      warning(sprintf("File not found: %s — skipping.", fpath))
      return(NULL)
    }
    
    df <- read_csv(fpath, show_col_types = FALSE)
    df[[TIME_VAR]] <- t   # Add numeric time variable
    df
  })
  
  if (nrow(all_data) == 0) stop("No data files were loaded. Check BASE_PATH and TIMEFRAMES.")
  all_data
}

cat("Loading data...\n")
raw_data <- load_data(TIMEFRAMES, BASE_PATH)
cat(sprintf("Loaded %d rows across %d timepoints.\n", nrow(raw_data), length(unique(raw_data[[TIME_VAR]]))))

# --- 4. Prepare Analysis Dataset ----------------------------------------------

analysis_data <- raw_data %>%
  select(all_of(c(ID_VAR, TIME_VAR, OUTCOME_VAR)),
         any_of(c("cbcl_q86_p", "cbcl_q87_p", "cbcl_q95_p"))) %>%
  filter(!is.na(.data[[OUTCOME_VAR]])) %>%
  # lcmm requires integer subject IDs
  mutate(subject_int = as.integer(factor(.data[[ID_VAR]])))

cat(sprintf("Unique subjects after cleaning: %d\n", n_distinct(analysis_data[[ID_VAR]])))
cat(sprintf("Timepoints represented: %s\n", paste(sort(unique(analysis_data[[TIME_VAR]])), collapse = ", ")))

# Quick descriptive summary
cat("\nOutcome variable summary by timepoint:\n")
analysis_data %>%
  group_by(.data[[TIME_VAR]]) %>%
  summarise(
    n       = n(),
    mean    = round(mean(.data[[OUTCOME_VAR]], na.rm = TRUE), 2),
    sd      = round(sd(.data[[OUTCOME_VAR]],   na.rm = TRUE), 2),
    median  = median(.data[[OUTCOME_VAR]],      na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print()

# --- 5. Fit LCGA Models (1 to MAX_CLASSES classes) ----------------------------
# LCGA is a constrained special case of LCMM where within-class variance
# of the random intercept is fixed to zero (mixture = FALSE in hlme).
# We use hlme() with fixed = outcome ~ time and random = ~1 (intercept only,
# constrained) to implement LCGA.

set.seed(SET_SEED)

fit_lcga <- function(data, n_classes, outcome, time, id) {
  formula_fixed  <- as.formula(sprintf("%s ~ %s", outcome, time))
  formula_random <- ~1  # Random intercept only (variance fixed to 0 in LCGA)
  
  cat(sprintf("\nFitting LCGA with %d class(es)...\n", n_classes))
  
  tryCatch({
    if (n_classes == 1) {
      # Single class: standard linear mixed model (no mixture)
      model <- hlme(
        fixed    = formula_fixed,
        random   = formula_random,
        subject  = "subject_int",
        ng       = 1,
        data     = as.data.frame(data),
        verbose  = FALSE
      )
    } else {
      # Multi-class: use 1-class model as starting values for stability
      model <- hlme(
        fixed    = formula_fixed,
        mixture  = formula_fixed,   # Class-specific trajectories (LCGA)
        random   = formula_random,
        subject  = "subject_int",
        ng       = n_classes,
        data     = as.data.frame(data),
        B        = models[[1]],     # Warm-start from 1-class solution
        nwg      = FALSE,           # Equal residual variances across classes (LCGA)
        verbose  = FALSE
      )
    }
    cat(sprintf("  Done. Log-likelihood: %.2f\n", model$loglik))
    model
  }, error = function(e) {
    warning(sprintf("Model with %d classes failed: %s", n_classes, e$message))
    NULL
  })
}

models <- vector("list", MAX_CLASSES)
for (k in seq_len(MAX_CLASSES)) {
  models[[k]] <- fit_lcga(analysis_data, k, OUTCOME_VAR, TIME_VAR, "subject_int")
}

# Remove failed models
models <- Filter(Negate(is.null), models)
cat(sprintf("\nSuccessfully fitted %d model(s).\n", length(models)))

# --- 6. Model Comparison Table ------------------------------------------------

# Helper: safely extract fit stats from an lcmm/hlme object
extract_fit <- function(m) {
  # safe_scalar: extract first non-NA numeric element or return NA
  safe_scalar <- function(x, as_fn = as.numeric) {
    tryCatch({
      v <- as_fn(x)
      v <- v[!is.na(v)]
      if (length(v) == 0) NA else v[[1]]
    }, error = function(e) NA)
  }

  ng     <- m$ng
  loglik <- safe_scalar(m$loglik)
  aic    <- safe_scalar(m$AIC)
  bic    <- safe_scalar(m$BIC)
  npm    <- safe_scalar(m$npm, as.integer)
  ns     <- safe_scalar(m$ns,  as.integer)

  # Fall back to summarytable() if direct fields are NULL/NA
  if (is.na(loglik)) {
    st <- tryCatch(summarytable(m), error = function(e) NULL)
    if (!is.null(st)) {
      loglik <- as.numeric(st[1, "loglik"])
      bic    <- as.numeric(st[1, "BIC"])
      aic    <- as.numeric(st[1, "AIC"])
    }
  }

  sabic <- tryCatch(
    round(-2 * loglik + log((ns + 2) / 24) * npm, 2),
    error   = function(e) NA_real_,
    warning = function(w) NA_real_
  )

  entropy       <- NA_real_
  min_classsize <- NA_integer_

  if (!is.null(m$pprob) && ng > 1) {
    prob_cols <- grep("^prob", colnames(m$pprob), value = TRUE)
    if (length(prob_cols) >= 2) {
      probs         <- as.matrix(m$pprob[, prob_cols, drop = FALSE])
      class_assign  <- apply(probs, 1, which.max)
      min_classsize <- as.integer(min(table(class_assign)))
      H             <- -sum(probs * log(probs + 1e-10))
      entropy       <- round(1 - H / (nrow(probs) * log(ncol(probs))), 3)
    }
  }

  tibble(
    Classes       = ng,
    LogLik        = round(loglik,  2),
    AIC           = round(aic,     2),
    BIC           = round(bic,     2),
    SABIC         = sabic,
    Entropy       = entropy,
    Min_ClassSize = min_classsize
  )
}

model_fit_table <- map_dfr(models, extract_fit)

cat("\n===== MODEL FIT COMPARISON =====\n")
print(model_fit_table)

# Recommend optimal number of classes (lowest BIC, minimum class size >= 5%)
n_subjects   <- n_distinct(analysis_data[[ID_VAR]])
min_size_thr <- ceiling(0.05 * n_subjects)

valid_models <- model_fit_table %>%
  filter(is.na(Min_ClassSize) | Min_ClassSize >= min_size_thr)

if (nrow(valid_models) == 0) {
  warning("No models meet the 5% min-class-size threshold; relaxing constraint.")
  valid_models <- model_fit_table
}

optimal_k     <- valid_models$Classes[which.min(valid_models$BIC)]
optimal_index <- which(sapply(models, function(m) m$ng) == optimal_k)[1]
cat(sprintf("\nRecommended number of classes (lowest BIC, min class >= 5%%): %d\n", optimal_k))

# --- 7. Extract Results from Optimal Model ------------------------------------

best_model <- models[[optimal_index]]


# --- 7a. Class membership probabilities --------------------------------------
class_probs <- best_model$pprob

# Identify the class-assignment column (lcmm names it "class")
# and probability columns (named "prob1", "prob2", ...)
class_id_col  <- "subject_int"   # first col is the integer subject id
class_val_col <- "class"         # lcmm stores MAP class here

# Rename subject column if needed
if (colnames(class_probs)[1] != "subject_int") {
  colnames(class_probs)[1] <- "subject_int"
}

# Derive assigned class from the "class" column if present, else from prob cols
if ("class" %in% colnames(class_probs)) {
  assigned_class <- class_probs$class
} else {
  prob_cols      <- grep("^prob", colnames(class_probs), value = TRUE)
  assigned_class <- apply(as.matrix(class_probs[, prob_cols, drop = FALSE]), 1, which.max)
}

# Merge back to original IDs
id_map <- analysis_data %>%
  distinct(subject_int, .data[[ID_VAR]])

class_assignments <- class_probs %>%
  mutate(assigned_class = assigned_class) %>%
  left_join(id_map, by = "subject_int") %>%
  relocate(all_of(ID_VAR), .before = everything())

cat(sprintf("\nClass distribution (optimal %d-class model):\n", optimal_k))
print(table(assigned_class))

# --- 8. Predict Trajectories for Plotting ------------------------------------

# predictY requires the outcome column to be present in newdata
time_grid <- data.frame(
  time_months          = seq(min(analysis_data[[TIME_VAR]]),
                             max(analysis_data[[TIME_VAR]]), by = 1),
  cbcl_irr_index_cnst  = 0,   # placeholder value; not used for fixed predictions
  subject_int          = analysis_data$subject_int[1]
)

pred_list <- tryCatch({
  predictY(best_model, newdata = time_grid, var.time = TIME_VAR, draws = FALSE)
}, error = function(e) {
  message("predictY failed (", conditionMessage(e), "); trajectories will use observed means.")
  NULL
})

# --- 9. Visualization ---------------------------------------------------------

# 9a. BIC Plot
p_bic <- ggplot(model_fit_table, aes(x = Classes, y = BIC)) +
  geom_line(color = "#2C7BB6", linewidth = 1) +
  geom_point(color = "#2C7BB6", size = 3) +
  geom_vline(xintercept = optimal_k, linetype = "dashed", color = "red", linewidth = 0.8) +
  annotate("text", x = optimal_k + 0.15, y = max(model_fit_table$BIC),
           label = sprintf("Optimal\nk = %d", optimal_k), color = "red", size = 3.5, hjust = 0) +
  labs(title = "Model Selection: BIC by Number of Classes",
       x = "Number of Latent Classes", y = "BIC (lower = better)") +
  theme_bw(base_size = 12)

# 9b. Trajectory Plot
# Build observed mean trajectories per assigned class
analysis_data_w_class <- analysis_data %>%
  left_join(
    tibble(subject_int = class_probs$subject_int,
           latent_class = factor(assigned_class)),
    by = "subject_int"
  )

traj_means <- analysis_data_w_class %>%
  group_by(latent_class, .data[[TIME_VAR]]) %>%
  summarise(mean_outcome = mean(.data[[OUTCOME_VAR]], na.rm = TRUE),
            se_outcome   = sd(.data[[OUTCOME_VAR]], na.rm = TRUE) / sqrt(n()),
            n = n(),
            .groups = "drop")

class_sizes <- analysis_data_w_class %>%
  distinct(subject_int, latent_class) %>%
  count(latent_class, name = "n_subjects")

traj_means <- traj_means %>%
  left_join(class_sizes, by = "latent_class") %>%
  mutate(class_label = sprintf("Class %s\n(n = %d)", latent_class, n_subjects))

p_traj <- ggplot(traj_means, aes(x = .data[[TIME_VAR]], y = mean_outcome,
                                  color = class_label, group = class_label)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  geom_ribbon(aes(ymin = mean_outcome - se_outcome,
                  ymax = mean_outcome + se_outcome,
                  fill = class_label), alpha = 0.15, color = NA) +
  scale_x_continuous(breaks = TIMEFRAMES,
                     labels = paste0(TIMEFRAMES, "m")) +
  labs(title    = sprintf("LCGA Trajectories (%d-Class Solution)", optimal_k),
       subtitle = "Mean ± SE of CBCL Irritability Index by Latent Class",
       x        = "Time (months)",
       y        = "CBCL Irritability Index",
       color    = "Latent Class",
       fill     = "Latent Class") +
  theme_bw(base_size = 12) +
  theme(legend.position = "right")

# 9c. Class size bar chart
p_size <- ggplot(class_sizes, aes(x = latent_class, y = n_subjects, fill = latent_class)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n_subjects,
                                100 * n_subjects / sum(n_subjects))),
            vjust = -0.3, size = 3.5) +
  labs(title = "Class Membership Sizes",
       x = "Latent Class", y = "Number of Subjects") +
  theme_bw(base_size = 12)

# Save all plots
# Output directory: defaults to an "lcga_output" folder next to the script.
# Change OUTPUT_DIR at the top of the script or set the env var LCGA_OUTPUT_DIR.
output_dir <- OUTPUT_DIR
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
cat(sprintf("Saving outputs to: %s\n", output_dir))

ggsave(file.path(output_dir, "lcga_bic_plot.png"),      p_bic,   width = 7,  height = 5,  dpi = 150)
ggsave(file.path(output_dir, "lcga_trajectories.png"),  p_traj,  width = 9,  height = 6,  dpi = 150)
ggsave(file.path(output_dir, "lcga_class_sizes.png"),   p_size,  width = 6,  height = 5,  dpi = 150)

cat("\nPlots saved to output directory.\n")

# --- 10. Save Outputs ---------------------------------------------------------

# Class assignments CSV
write_csv(class_assignments,
          file.path(output_dir, "lcga_class_assignments.csv"))

# Model fit table CSV
write_csv(model_fit_table,
          file.path(output_dir, "lcga_model_fit_table.csv"))

cat("Class assignments saved to: lcga_class_assignments.csv\n")
cat("Model fit table saved to:   lcga_model_fit_table.csv\n")

# --- 11. Print Summary --------------------------------------------------------

cat("\n===== OPTIMAL MODEL SUMMARY =====\n")
summary(best_model)

cat("\n===== SCRIPT COMPLETE =====\n")
cat(sprintf("Optimal solution: %d latent classes\n", optimal_k))
cat(sprintf("BIC: %.2f | AIC: %.2f\n", 
            model_fit_table$BIC[model_fit_table$Classes == optimal_k],
            model_fit_table$AIC[model_fit_table$Classes == optimal_k]))