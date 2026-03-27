# ==============================================================================
# LCGA Class Characterization
# Compares baseline demographics and clinical variables across the 3 irritability
# trajectory classes identified by the LCGA script.
#
# Inputs:
#   1. lcga_class_assignments.csv  — output from lcga_abcd_irritability.R
#   2. pdem02.txt                  — ABCD demographics
#   3. abcd_cbcl01.txt             — CBCL syndrome + broadband scores
#   4. abcd_cbcls01.txt            — CBCL DSM-oriented scales
#   5. abcd_ksad01.txt             — K-SADS parent diagnostic interview
#   6. diff_emotion_reg_p01.txt    — DERS parent-report emotion dysregulation
#   7. opp_defiant_disorder_p01.txt   — ODD symptom items (K-SADS)
#
# Outputs:
#   - characterization_summary.csv     : formatted Table 2 (means/% + p-values)
#   - characterization_pairwise.csv    : FDR-corrected pairwise comparisons
#   - characterization_plots.png       : bar/violin plots per variable
# ==============================================================================

# --- 1. Packages --------------------------------------------------------------

required_packages <- c("tidyverse", "data.table",
                       "ggplot2", "gridExtra", "RColorBrewer", "dunn.test")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}
invisible(lapply(required_packages, install_if_missing))

library(tidyverse)
library(data.table)
library(ggplot2)
library(RColorBrewer)
library(dunn.test)

# --- 2. Configuration ---------------------------------------------------------

# Path to ABCD Package_1215452 folder
ABCD_PATH   <- Sys.getenv("ABCD_PATH",
                           unset = "/shared/healthinfolab/datasets/ABCD/Package_1215452")

# Path to class assignments from LCGA script
CLASS_FILE  <- Sys.getenv("LCGA_CLASS_FILE",
                           unset = file.path(getwd(), "lcga_output",
                                             "lcga_class_assignments.csv"))

OUTPUT_DIR  <- Sys.getenv("LCGA_OUTPUT_DIR",
                           unset = file.path(getwd(), "lcga_output"))

BASELINE    <- "baseline_year_1_arm_1"   # ABCD eventname for baseline

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 3. Load Class Assignments ------------------------------------------------

cat("Loading class assignments...\n")
classes <- fread(CLASS_FILE)[, .(src_subject_id, latent_class = as.integer(class))]
classes[, latent_class := factor(latent_class,
                                  levels = 1:3,
                                  labels = c("Class 1\nLow-Stable",
                                             "Class 2\nHigh-Decreasing",
                                             "Class 3\nLow-Increasing"))]
cat(sprintf("  %d subjects with class assignments.\n", nrow(classes)))
print(table(classes$latent_class))

# --- 4. Helper: safe file loader ---------------------------------------------
# ABCD .txt files are tab-separated with a second descriptor row that must
# be skipped.

load_abcd <- function(filename, abcd_path, eventname = BASELINE,
                      cols = NULL) {
  fpath <- file.path(abcd_path, filename)
  if (!file.exists(fpath)) {
    warning(sprintf("File not found: %s", fpath))
    return(NULL)
  }
  # Read true column names from row 1.
  col_names <- names(fread(fpath, nrows = 0))

  # Auto-detect how many header rows to skip.
  # Most ABCD files have 2 (col names + descriptor). Some only have 1.
  # If row 2's first field equals the first col name, it is another header row.
  row2_val <- as.character(fread(fpath, skip = 1, nrows = 1, header = FALSE)[[1]])
  n_skip   <- if (row2_val == col_names[1]) 2L else 1L

  dt <- fread(fpath, skip = n_skip, col.names = col_names,
              na.strings = c("", "NA", "999", "777", "888", "999.0"))

  # Filter to requested eventname. Use $ to avoid data.table NSE name clash.
  if (!is.null(eventname) && "eventname" %in% names(dt)) {
    ev <- eventname
    dt <- dt[dt$eventname == ev]
  }

  # Select only requested columns (always keep IDs)
  id_cols <- intersect(c("src_subject_id", "eventname"), names(dt))
  if (!is.null(cols)) {
    keep <- union(id_cols, intersect(cols, names(dt)))
    dt   <- dt[, ..keep]
  }

  cat(sprintf("  Loaded %-30s  %d rows, %d cols  [skip=%d]\n",
              filename, nrow(dt), ncol(dt), n_skip))
  dt
}

# --- 5. Load & Prepare Each Data Source --------------------------------------

cat("\nLoading ABCD data files...\n")

# ---- 5a. Demographics (pdem02) ----------------------------------------------
# Race is stored as binary indicator columns (demo_race_a_p___10 to ___25)
# rather than a single variable — we derive a single race_eth variable from
# these plus the Hispanic ethnicity item (demo_ethn_v2).

dem <- load_abcd("pdem02.txt", ABCD_PATH)   # Load all columns

if (!is.null(dem)) {
  # --- Age ---
  dem[, age_years := as.numeric(interview_age) / 12]

  # --- Sex (demo_sex_v2: 1=Male, 2=Female, 3=Intersex/Other) ---
  dem[, sex := factor(fcase(
    as.integer(demo_sex_v2) == 1, "Male",
    as.integer(demo_sex_v2) == 2, "Female",
    as.integer(demo_sex_v2) == 3, "Intersex-Male",
    as.integer(demo_sex_v2) == 4, "Intersex-Female",
    default = NA_character_
  ), levels = c("Male", "Female", "Intersex-Male", "Intersex-Female"))]

  # --- Race/ethnicity ---
  # Derived using ABCD's standard 5-category scheme:
  #   Hispanic (any race)  > Black  > Asian  > White  > Other/Multiracial
  # Binary race columns: 10=White 11=Black 18-24=Asian 25=Other
  # Hispanic ethnicity:  demo_ethn_v2 == 1

  to_int <- function(x) suppressWarnings(as.integer(x))

  dem[, race_eth := fcase(
    to_int(demo_ethn_v2) == 1,                                 "Hispanic",
    to_int(demo_race_a_p___11) == 1,                           "Black",
    to_int(demo_race_a_p___18) == 1 | to_int(demo_race_a_p___19) == 1 |
      to_int(demo_race_a_p___20) == 1 | to_int(demo_race_a_p___21) == 1 |
      to_int(demo_race_a_p___22) == 1 | to_int(demo_race_a_p___23) == 1 |
      to_int(demo_race_a_p___24) == 1,                         "Asian",
    to_int(demo_race_a_p___10) == 1,                           "White",
    default =                                                   "Other/Multiracial"
  )]
  dem[, race_eth := factor(race_eth,
                            levels = c("White", "Black", "Hispanic",
                                       "Asian", "Other/Multiracial"))]

  # --- Family income (demo_comb_income_v2: 1-10 scale) ---
  dem[, income_group := fcase(
    as.integer(demo_comb_income_v2) %in% 1:6,  "< $50k",
    as.integer(demo_comb_income_v2) %in% 7:8,  "$50k-$99k",
    as.integer(demo_comb_income_v2) %in% 9:10, ">= $100k"
  )]
  dem[, income_group := factor(income_group,
                                levels = c("< $50k", "$50k-$99k", ">= $100k"))]

  # --- Caregiver education (demo_prnt_ed_v2: 0-21 scale) ---
  dem[, parent_edu := fcase(
    as.integer(demo_prnt_ed_v2) %in% 0:12,  "No college",
    as.integer(demo_prnt_ed_v2) %in% 13:14, "Some college",
    as.integer(demo_prnt_ed_v2) %in% 15:21, "College+"
  )]
  dem[, parent_edu := factor(parent_edu,
                              levels = c("No college", "Some college", "College+"))]

  # --- Marital status (demo_prnt_marital_v2: 1=Married) ---
  dem[, married := factor(
    fifelse(as.integer(demo_prnt_marital_v2) == 1, "Married", "Not married"),
    levels = c("Married", "Not married")
  )]

  dem <- dem[, .(src_subject_id, age_years, sex, race_eth,
                 income_group, parent_edu, married)]
}

# ---- 5b+5c. CBCL syndrome + DSM-oriented scores (both in abcd_cbcls01) ------
# NOTE: cbcl_scr_syn_* T-scores are NOT in abcd_cbcl01.txt in this package —
# they are in abcd_cbcls01.txt alongside the DSM-oriented scales.
cbcl_cols <- c(
  "cbcl_scr_syn_anxdep_t",      # Anxious/Depressed T-score
  "cbcl_scr_syn_withdep_t",     # Withdrawn/Depressed T-score
  "cbcl_scr_syn_somatic_t",     # Somatic Complaints T-score
  "cbcl_scr_syn_social_t",      # Social Problems T-score
  "cbcl_scr_syn_thought_t",     # Thought Problems T-score
  "cbcl_scr_syn_attention_t",   # Attention Problems T-score
  "cbcl_scr_syn_rulebreak_t",   # Rule-Breaking T-score
  "cbcl_scr_syn_aggressive_t",  # Aggressive Behavior T-score
  "cbcl_scr_syn_internal_t",    # Internalizing T-score
  "cbcl_scr_syn_external_t",    # Externalizing T-score
  "cbcl_scr_syn_totprob_t",     # Total Problems T-score
  "cbcl_scr_dsm5_adhd_t",       # ADHD DSM5 T-score
  "cbcl_scr_dsm5_depress_t",    # Depressive Problems T-score
  "cbcl_scr_dsm5_anxdisord_t",  # Anxiety Problems T-score
  "cbcl_scr_dsm5_opposit_t",    # Oppositional Defiant T-score
  "cbcl_scr_dsm5_conduct_t"     # Conduct Problems T-score
)

cbcl  <- load_abcd("abcd_cbcls01.txt", ABCD_PATH, cols = cbcl_cols)
cbcls <- NULL   # all scores now loaded in cbcl above

# ---- 5d. K-SADS ADHD + Irritability (abcd_ksad01) --------------------------
# ADHD: sum of 18 symptom items (binary 0/1)
# Irritability: 4 items identified by Lee et al. (2022, Front. Psychiatry)
#   as the core transdiagnostic irritability indicators in the ABCD K-SADS:
#     ksads_1_3_p   — Irritability Present (MDD module)
#     ksads_3_229_p — Temper/irritability present in at least 2 settings (DMDD module)
#     ksads_15_432_p — Often touchy or easily annoyed Present (ODD module)
#     ksads_15_91_p  — Often loses temper Present (ODD module)

ksad <- load_abcd("abcd_ksad01.txt", ABCD_PATH)

if (!is.null(ksad)) {

  # --- ADHD symptom count ---
  adhd_items <- grep("^ksads_14_", names(ksad), value = TRUE)
  if (length(adhd_items) > 0) {
    clamp_binary <- function(x) { x <- suppressWarnings(as.numeric(x)); x[!x %in% c(0,1)] <- NA; x }
    ksad[, (adhd_items) := lapply(.SD, clamp_binary), .SDcols = adhd_items]
    ksad[, adhd_symptom_count := rowSums(.SD, na.rm = TRUE),
         .SDcols = adhd_items]
  } else {
    warning("No ADHD items found in abcd_ksad01.txt — check column names.")
    ksad[, adhd_symptom_count := NA_real_]
  }

  # --- K-SADS irritability sum (0-4 scale) ---
  irr_items <- c("ksads_1_3_p", "ksads_3_229_p",
                 "ksads_15_432_p", "ksads_15_91_p")
  irr_present <- intersect(irr_items, names(ksad))

  if (length(irr_present) > 0) {
    # Clamp to binary (0/1) — codes like 888 become NA
    clamp_binary <- function(x) { x <- suppressWarnings(as.numeric(x)); x[!x %in% c(0,1)] <- NA; x }
    ksad[, (irr_present) := lapply(.SD, clamp_binary), .SDcols = irr_present]
    ksad[, ksads_irritability_sum := rowSums(.SD, na.rm = TRUE),
         .SDcols = irr_present]
    cat(sprintf("  K-SADS irritability: %d of 4 items found (%s)
",
                length(irr_present), paste(irr_present, collapse = ", ")))
  } else {
    warning("No K-SADS irritability items found — ksads_irritability_sum will be NA.")
    ksad[, ksads_irritability_sum := NA_real_]
  }

  ksad <- ksad[, .(src_subject_id, adhd_symptom_count, ksads_irritability_sum)]
}

# ---- 5e. DERS — Difficulties in Emotion Regulation (diff_emotion_reg_p01) ---
# This file has no baseline rows — it was collected at follow-up only.
# We load all timepoints (eventname = NULL) and take each subject's earliest
# available observation so we have one row per subject.

ders_raw <- load_abcd("diff_emotion_reg_p01.txt", ABCD_PATH, eventname = NULL)

if (!is.null(ders_raw) && nrow(ders_raw) > 0) {
  upset_items <- grep("^ders_upset_", names(ders_raw), value = TRUE)

  if (length(upset_items) > 0) {
    # Coerce to numeric and clamp to valid Likert range (1-5); 777 -> NA
    clamp15 <- function(x) { x <- suppressWarnings(as.numeric(x)); x[x < 1 | x > 5] <- NA; x }
    ders_raw[, (upset_items) := lapply(.SD, clamp15), .SDcols = upset_items]

    ders_raw[, ders_total := rowSums(.SD, na.rm = TRUE), .SDcols = upset_items]

    if ("ders_upset_irritation_p" %in% names(ders_raw)) {
      ders_raw[, ders_irritation := clamp15(ders_upset_irritation_p)]
    } else {
      ders_raw[, ders_irritation := NA_real_]
      warning("ders_upset_irritation_p not found.")
    }

    # One row per subject: keep earliest timepoint
    # interview_age is in months — lower = earlier
    ders_raw[, interview_age := suppressWarnings(as.numeric(interview_age))]
    setorder(ders_raw, src_subject_id, interview_age)
    ders <- unique(ders_raw[, .(src_subject_id, ders_total, ders_irritation)],
                   by = "src_subject_id")

    cat(sprintf("  DERS: %d subjects with data (earliest timepoint used)\n", nrow(ders)))
  } else {
    warning("No ders_upset_ items found in diff_emotion_reg_p01.txt.")
    ders <- NULL
  }
} else {
  warning("diff_emotion_reg_p01.txt loaded 0 rows — DERS will be excluded.")
  ders <- NULL
}

# ---- 5f. ODD — Oppositional Defiant Disorder (opp_defiant_disorder_p01) -----
# The file contains a mix of binary symptom items (0/1) and free-text duration
# fields (e.g. "weeks:0months:3years:0"). We sum only the strictly binary
# columns (all values in {0, 1, NA}) to get a clean symptom count.

odd_raw <- load_abcd("opp_defiant_disorder_p01.txt", ABCD_PATH)

if (!is.null(odd_raw) && nrow(odd_raw) > 0) {
  odd_candidates <- grep("^ksads_odd_raw_", names(odd_raw), value = TRUE)

  # Keep only columns whose non-NA values are all 0 or 1
  is_binary <- function(col) {
    vals <- suppressWarnings(as.numeric(odd_raw[[col]]))
    all(is.na(vals) | vals %in% c(0, 1))
  }
  odd_items <- Filter(is_binary, odd_candidates)

  cat(sprintf("  ODD: %d binary items retained out of %d total\n",
              length(odd_items), length(odd_candidates)))

  if (length(odd_items) > 0) {
    odd_raw[, (odd_items) := lapply(.SD, function(x) suppressWarnings(as.numeric(x))),
            .SDcols = odd_items]
    odd_raw[, odd_symptom_count := rowSums(.SD, na.rm = TRUE), .SDcols = odd_items]
    odd <- odd_raw[, .(src_subject_id, odd_symptom_count)]
  } else {
    warning("No binary ODD items found — odd_symptom_count will be excluded.")
    odd <- NULL
  }
} else {
  odd <- NULL
}

# --- 6. Merge Everything with Class Assignments -------------------------------

cat("\nMerging data...\n")

merged <- classes
for (dt in list(dem, cbcl, cbcls, ksad, ders, odd)) {
  if (!is.null(dt))
    merged <- merge(merged, dt, by = "src_subject_id", all.x = TRUE)
}

cat(sprintf("Final merged dataset: %d subjects, %d variables\n",
            nrow(merged), ncol(merged)))

# --- 7. Define Variable Lists for Analysis ------------------------------------

# Continuous variables: Kruskal-Wallis + Dunn pairwise
continuous_vars <- c(
    "age_years",
    "cbcl_scr_syn_anxdep_t", "cbcl_scr_syn_withdep_t", "cbcl_scr_syn_somatic_t",
    "cbcl_scr_syn_social_t",  "cbcl_scr_syn_thought_t", "cbcl_scr_syn_attention_t",
    "cbcl_scr_syn_rulebreak_t", "cbcl_scr_syn_aggressive_t",
    "cbcl_scr_syn_internal_t", "cbcl_scr_syn_external_t",
    "cbcl_scr_syn_totprob_t",
    "cbcl_scr_dsm5_adhd_t", "cbcl_scr_dsm5_depress_t",
    "cbcl_scr_dsm5_anxdisord_t", "cbcl_scr_dsm5_opposit_t",
    "cbcl_scr_dsm5_conduct_t",
    "adhd_symptom_count",
    "ksads_irritability_sum", # K-SADS 4-item irritability score (Lee et al., 2022)
    "ders_total",        # DERS total emotion dysregulation
    "ders_irritation",   # DERS irritation item (single, directly taps irritability)
    "odd_symptom_count"  # ODD binary symptom count
)

# Categorical variables: Chi-square
categorical_vars <- c("sex", "race_eth", "income_group", "parent_edu", "married")

# Keep only variables that exist in merged AND have at least some non-NA data
has_data <- function(var) {
  var %in% names(merged) && sum(!is.na(merged[[var]])) > 0
}

continuous_vars  <- Filter(has_data, continuous_vars)
categorical_vars <- Filter(has_data, categorical_vars)

cat(sprintf("\nContinuous variables to test:  %d\n", length(continuous_vars)))
cat(sprintf("Categorical variables to test: %d\n", length(categorical_vars)))

# --- 8. Statistical Tests -----------------------------------------------------

# ---- 8a. Continuous: Kruskal-Wallis + Dunn post-hoc -------------------------
cat("\nRunning Kruskal-Wallis tests...\n")

kw_results <- lapply(continuous_vars, function(var) {

  df <- merged[!is.na(get(var)), .(value = get(var), latent_class)]

  if (length(unique(df$latent_class)) < 2) {
    warning(sprintf("Skipping %s: only one class present after filtering.", var))
    return(NULL)
  }

  kw <- kruskal.test(value ~ latent_class, data = df)

  desc <- df[, .(
    mean = round(mean(value, na.rm = TRUE), 2),
    sd   = round(sd(value,   na.rm = TRUE), 2),
    n    = sum(!is.na(value))
  ), by = latent_class]

  list(var = var, kw_p = kw$p.value, desc = desc)
})

# Remove NULL results
kw_results <- Filter(Negate(is.null), kw_results)

# Dunn post-hoc for significant KW results (before FDR)
cat("Running Dunn pairwise tests...\n")

dunn_results <- lapply(kw_results, function(res) {

  if (is.null(res)) return(NULL)

  df <- merged[!is.na(get(res$var)), .(value = get(res$var), latent_class)]

  if (length(unique(df$latent_class)) < 2) return(NULL)

  out <- dunn.test(df$value, df$latent_class, method = "none", kw = FALSE,
                   label = TRUE, wrap = FALSE, table = FALSE, list = FALSE,
                   rmc = FALSE, alpha = 0.05, altp = FALSE)

  tibble(
    variable    = res$var,
    comparison  = out$comparisons,
    Z           = round(out$Z, 3),
    p_raw       = out$P
  )
})

dunn_df <- bind_rows(dunn_results)

dunn_df$p_fdr <- p.adjust(dunn_df$p_raw, method = "fdr")
dunn_df$sig   <- case_when(
  dunn_df$p_fdr < 0.001 ~ "***",
  dunn_df$p_fdr < 0.01  ~ "**",
  dunn_df$p_fdr < 0.05  ~ "*",
  TRUE                   ~ "ns"
)

# ---- 8b. Categorical: Chi-square --------------------------------------------
cat("Running chi-square tests...\n")

chi_results <- lapply(categorical_vars, function(var) {
  tab <- table(merged[[var]], merged$latent_class, useNA = "no")
  chi <- tryCatch(chisq.test(tab), error = function(e) NULL)
  list(
    var   = var,
    chi_p = if (!is.null(chi)) chi$p.value else NA,
    table = as.data.frame(prop.table(tab, margin = 2) * 100)
  )
})

# FDR correction across all omnibus tests
all_p   <- c(sapply(kw_results,  `[[`, "kw_p"),
             sapply(chi_results, `[[`, "chi_p"))
all_fdr <- p.adjust(all_p, method = "fdr")

# --- 9. Build Summary Table ---------------------------------------------------
cat("\nBuilding summary table...\n")

# Continuous rows: "Mean (SD)" per class + KW p-value (FDR-corrected)
n_cont <- length(continuous_vars)
cont_rows <- lapply(seq_along(kw_results), function(i) {
  res  <- kw_results[[i]]
  desc <- res$desc
  row  <- tibble(Variable = res$var, Type = "Continuous")

  for (cls in levels(merged$latent_class)) {
    d <- desc[latent_class == cls]
    row[[cls]] <- if (nrow(d) > 0)
      sprintf("%.2f (%.2f)", d$mean, d$sd) else "—"
  }
  row$p_fdr <- round(all_fdr[i], 4)
  row$sig   <- case_when(
    row$p_fdr < 0.001 ~ "***",
    row$p_fdr < 0.01  ~ "**",
    row$p_fdr < 0.05  ~ "*",
    TRUE               ~ "ns"
  )
  row
})

# Categorical rows: "n (%)" per class + chi-square p-value (FDR-corrected)
cat_rows <- lapply(seq_along(chi_results), function(i) {
  res <- chi_results[[i]]
  tab <- as.data.table(res$table)
  setnames(tab, c("level", "latent_class", "pct"))
  counts <- as.data.table(table(merged[[res$var]], merged$latent_class,
                                useNA = "no"))
  setnames(counts, c("level", "latent_class", "n"))
  tab <- merge(tab, counts, by = c("level", "latent_class"))

  lapply(unique(tab$level), function(lvl) {
    row <- tibble(Variable = sprintf("%s: %s", res$var, lvl),
                  Type = "Categorical")
    for (cls in levels(merged$latent_class)) {
      d <- tab[level == lvl & latent_class == cls]
      row[[cls]] <- if (nrow(d) > 0)
        sprintf("%d (%.1f%%)", d$n, d$pct) else "—"
    }
    row$p_fdr <- round(all_fdr[n_cont + i], 4)
    row$sig   <- case_when(
      row$p_fdr < 0.001 ~ "***",
      row$p_fdr < 0.01  ~ "**",
      row$p_fdr < 0.05  ~ "*",
      TRUE               ~ "ns"
    )
    row
  })
})

summary_table <- bind_rows(c(cont_rows, unlist(cat_rows, recursive = FALSE)))

cat("\n===== CHARACTERIZATION SUMMARY =====\n")
print(summary_table, n = Inf)

# --- 10. Visualization --------------------------------------------------------
cat("\nGenerating plots...\n")

class_palette <- c(
  "Class 1\nLow-Stable"       = "#4f8bc8",
  "Class 2\nHigh-Decreasing"  = "#8b3fca",
  "Class 3\nLow-Increasing"   = "#cb6587"
)

# Violin + boxplot for key variables
plot_vars <- intersect(
  c(
    "cbcl_scr_syn_totprob_t", "cbcl_scr_syn_external_t",
    "cbcl_scr_syn_internal_t", "cbcl_scr_syn_aggressive_t",
    "cbcl_scr_syn_attention_t", "cbcl_scr_dsm5_adhd_t",
    "cbcl_scr_dsm5_opposit_t",  "cbcl_scr_dsm5_depress_t",
    "ksads_irritability_sum",
    "ders_total",
    "ders_irritation",
    "odd_symptom_count"
  ),
  names(merged)
)

var_labels <- c(
    cbcl_scr_syn_totprob_t    = "Total Problems",
    cbcl_scr_syn_external_t   = "Externalizing",
    cbcl_scr_syn_internal_t   = "Internalizing",
    cbcl_scr_syn_aggressive_t = "Aggressive Behavior",
    cbcl_scr_syn_attention_t  = "Attention Problems",
    cbcl_scr_dsm5_adhd_t      = "ADHD (DSM5)",
    cbcl_scr_dsm5_opposit_t   = "ODD (DSM5)",
    cbcl_scr_dsm5_depress_t   = "Depression (DSM5)",
    ksads_irritability_sum    = "Irritability Score (K-SADS, 0-4)",
    ders_total                = "Emotion Dysregulation (DERS)",
    ders_irritation           = "Irritation Item (DERS)",
    odd_symptom_count         = "ODD Symptoms"
)

violin_plots <- lapply(plot_vars, function(var) {
  df_plot <- as.data.frame(merged[!is.na(get(var)),
                                   .(latent_class, value = get(var))])

  # Skip variables with no data or only one class represented
  if (nrow(df_plot) < 10 || length(unique(df_plot$latent_class)) < 2) {
    warning(sprintf("Skipping plot for '%s': insufficient data across classes.", var))
    return(NULL)
  }

  label <- ifelse(var %in% names(var_labels), var_labels[[var]], var)

  # Informative y-axis label and optional limits per variable type
  cbcl_t_vars <- c("cbcl_scr_syn_totprob_t", "cbcl_scr_syn_external_t",
                   "cbcl_scr_syn_internal_t", "cbcl_scr_syn_aggressive_t",
                   "cbcl_scr_syn_attention_t", "cbcl_scr_dsm5_adhd_t",
                   "cbcl_scr_dsm5_opposit_t", "cbcl_scr_dsm5_depress_t")

  y_label <- dplyr::case_when(
    var %in% cbcl_t_vars             ~ "T-score (mean=50, SD=10)",
    var == "ksads_irritability_sum"  ~ "Items endorsed (0=none, 4=all)",
    var == "ders_total"              ~ "Sum score (20 items, 1\u20135 scale)",
    var == "ders_irritation"         ~ "Rating (1=Never, 5=Always)",
    var == "odd_symptom_count"       ~ "Symptoms endorsed (out of 25)",
    TRUE                             ~ "Value"
  )

  y_limits <- dplyr::case_when(
    var == "ders_irritation"        ~ list(c(1, 5)),
    var == "ksads_irritability_sum" ~ list(c(0, 4)),
    TRUE                            ~ list(NULL)
  )[[1]]

  p <- ggplot(df_plot, aes(x = latent_class, y = value, fill = latent_class)) +
    geom_violin(trim = TRUE, alpha = 0.6, color = NA) +
    geom_boxplot(width = 0.15, outlier.size = 0.3,
                 fill = "white", color = "grey30") +
    scale_fill_manual(values = class_palette) +
    labs(title = label, x = NULL, y = y_label) +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          axis.text.x = element_text(size = 8))

  if (!is.null(y_limits))
    p <- p + coord_cartesian(ylim = y_limits)

  p
})

# Remove NULL plots before arranging — prevents empty/black filler panels
violin_plots <- Filter(Negate(is.null), violin_plots)

n_plots  <- length(violin_plots)
n_cols   <- 4L
n_rows   <- ceiling(n_plots / n_cols)

combined_plot <- gridExtra::arrangeGrob(grobs = violin_plots, ncol = n_cols, nrow = n_rows)

ggsave(file.path(OUTPUT_DIR, "characterization_cbcl_plots.png"),
       combined_plot,
       width = 14, height = 4 * n_rows, dpi = 150)

# Demographic bar charts (categorical)
demo_plots <- lapply(categorical_vars, function(var) {
  df_plot <- as.data.frame(merged[!is.na(get(var)),
                                   .(latent_class, value = get(var))])
  df_pct  <- df_plot %>%
    count(latent_class, value) %>%
    group_by(latent_class) %>%
    mutate(pct = 100 * n / sum(n)) %>%
    ungroup()

  ggplot(df_pct, aes(x = value, y = pct, fill = latent_class)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_manual(values = class_palette, name = "Class") +
    labs(title = var, x = NULL, y = "% within class") +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9))
})

demo_combined <- gridExtra::arrangeGrob(grobs = demo_plots, ncol = 3, nrow = ceiling(length(demo_plots) / 3))

ggsave(file.path(OUTPUT_DIR, "characterization_demo_plots.png"),
       demo_combined,
       width = 14, height = 5 * ceiling(length(demo_plots) / 3), dpi = 150)

cat("Plots saved.\n")

# --- 11. Save Results ---------------------------------------------------------

fwrite(summary_table,
       file.path(OUTPUT_DIR, "characterization_summary.csv"))

fwrite(dunn_df,
       file.path(OUTPUT_DIR, "characterization_pairwise.csv"))

cat("\nSaved:\n")
cat(sprintf("  %s/characterization_summary.csv\n",  OUTPUT_DIR))
cat(sprintf("  %s/characterization_pairwise.csv\n", OUTPUT_DIR))
cat(sprintf("  %s/characterization_cbcl_plots.png\n", OUTPUT_DIR))
cat(sprintf("  %s/characterization_demo_plots.png\n", OUTPUT_DIR))
cat("\n===== CHARACTERIZATION COMPLETE =====\n")