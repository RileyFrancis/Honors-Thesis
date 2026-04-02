# ==============================================================================
# Count subjects plotted per variable per class
# Mirrors the 12 variables shown in characterization_cbcl_plots.png
# ==============================================================================

library(tidyverse)
library(data.table)

# --- Configuration ------------------------------------------------------------

CLASS_FILE <- Sys.getenv("LCGA_CLASS_FILE",
                          unset = file.path(getwd(), "lcga_output",
                                            "lcga_class_assignments.csv"))
OUTPUT_DIR <- Sys.getenv("LCGA_OUTPUT_DIR",
                          unset = file.path(getwd(), "lcga_output"))
ABCD_PATH  <- Sys.getenv("ABCD_PATH",
                          unset = "/shared/healthinfolab/datasets/ABCD/Package_1215452")
BASELINE   <- "baseline_year_1_arm_1"

# --- Helper -------------------------------------------------------------------

load_abcd <- function(filename, cols = NULL) {
  fpath <- file.path(ABCD_PATH, filename)
  if (!file.exists(fpath)) { warning(sprintf("Not found: %s", fpath)); return(NULL) }
  col_names <- names(fread(fpath, nrows = 0))
  dt <- fread(fpath, skip = 2, col.names = col_names,
              na.strings = c("", "NA", "999", "777", "999.0"))
  if ("eventname" %in% names(dt)) dt <- dt[eventname == BASELINE]
  id_cols <- intersect(c("src_subject_id", "eventname"), names(dt))
  if (!is.null(cols)) {
    keep <- union(id_cols, intersect(cols, names(dt)))
    dt   <- dt[, ..keep]
  }
  dt
}

# --- Load class assignments ---------------------------------------------------

classes <- fread(CLASS_FILE)[, .(
  src_subject_id,
  latent_class = factor(as.integer(class),
                        levels = 1:3,
                        labels = c("Class 1\nLow-Stable",
                                   "Class 2\nHigh-Decreasing",
                                   "Class 3\nLow-Increasing"))
)]

# --- Load each data source with just the plotted variables --------------------

# ALL CBCL scores are in abcd_cbcls01 (abcd_cbcl01 has raw items only)
cbcls <- load_abcd("abcd_cbcls01.txt", cols = c(
  "cbcl_scr_syn_totprob_t",
  "cbcl_scr_syn_external_t",
  "cbcl_scr_syn_internal_t",
  "cbcl_scr_syn_aggressive_t",
  "cbcl_scr_syn_attention_t",
  "cbcl_scr_dsm5_adhd_t",
  "cbcl_scr_dsm5_depress_t",
  "cbcl_scr_dsm5_opposit_t"
))

# K-SADS irritability: use live items confirmed by diagnostic
# ksads_1_3_p  = irritable mood (current episode, 228 endorsements)
# ksads_1_4_p  = irritable mood (past episode,   927 endorsements)
# ksads_1_156_p / ksads_1_164_p = additional irritability items
# Value 888 = "not applicable/skipped" — treat as NA via na.strings above
ksad_irr_cols <- c("ksads_1_3_p", "ksads_1_4_p",
                   "ksads_1_156_p", "ksads_1_164_p")
ksad <- load_abcd("abcd_ksad01.txt", cols = ksad_irr_cols)
if (!is.null(ksad)) {
  # Re-code 888 as NA (not caught by na.strings because it appears as integer)
  for (col in ksad_irr_cols) {
    if (col %in% names(ksad)) set(ksad, which(ksad[[col]] == 888L), col, NA_integer_)
  }
  present_irr <- intersect(ksad_irr_cols, names(ksad))
  ksad[, irritability_score := rowSums(.SD, na.rm = FALSE), .SDcols = present_irr]
  ksad <- ksad[, .(src_subject_id, irritability_score)]
}

# Emotion dysregulation: DERS files exist but are EMPTY at baseline in ABCD.
# Best available substitute is the PGBI (General Behavior Inventory, parent),
# which has full baseline coverage and measures mood/emotion dysregulation.
# Items: gen_child_behav_1 to gen_child_behav_10 (0/1 binary, summed 0-10)
pgbi_cols <- paste0("gen_child_behav_", 1:10)
pgbi <- load_abcd("abcd_pgbi01.txt", cols = pgbi_cols)
if (!is.null(pgbi)) {
  present_pgbi <- intersect(pgbi_cols, names(pgbi))
  pgbi[, pgbi_total := rowSums(.SD, na.rm = FALSE), .SDcols = present_pgbi]
  pgbi <- pgbi[, .(src_subject_id, pgbi_total)]
  cat(sprintf("  PGBI: %d subjects, mean=%.2f, sd=%.2f
",
              sum(!is.na(pgbi$pgbi_total)),
              mean(pgbi$pgbi_total, na.rm = TRUE),
              sd(pgbi$pgbi_total,   na.rm = TRUE)))
}

# DMDD file: ksads_dmdd_raw_961_p through ksads_dmdd_raw_972_p
# Check if any items have non-missing data
dmdd_cols <- paste0("ksads_dmdd_raw_", c(961:972), "_p")
dmdd <- load_abcd("disruptive_mood_dysreg_p01.txt", cols = dmdd_cols)
if (!is.null(dmdd)) {
  present_dmdd <- intersect(dmdd_cols, names(dmdd))
  if (length(present_dmdd) > 0) {
    # Coerce all columns to integer (some may be read as character/IDat)
    dmdd[, (present_dmdd) := lapply(.SD, function(x) suppressWarnings(as.integer(x))),
         .SDcols = present_dmdd]
    dmdd[, dmdd_symptom_count := rowSums(.SD, na.rm = FALSE), .SDcols = present_dmdd]
    n_nonmiss <- sum(!is.na(dmdd$dmdd_symptom_count))
    cat(sprintf("  DMDD: %d subjects with non-missing data
", n_nonmiss))
    dmdd <- dmdd[, .(src_subject_id, dmdd_symptom_count)]
  } else {
    dmdd <- NULL
  }
}

# ODD symptoms: ksads_2_830-839 = ODD symptom items in ABCD
odd_cols_use <- c("ksads_2_830_p", "ksads_2_831_p", "ksads_2_832_p",
                  "ksads_2_833_p", "ksads_2_834_p", "ksads_2_835_p",
                  "ksads_2_836_p", "ksads_2_837_p", "ksads_2_838_p",
                  "ksads_2_839_p")
odd <- load_abcd("abcd_ksad01.txt", cols = odd_cols_use)
if (!is.null(odd)) {
  present_odd <- intersect(odd_cols_use, names(odd))
  if (length(present_odd) > 0) {
    odd[, odd_symptom_count := rowSums(.SD, na.rm = FALSE), .SDcols = present_odd]
    odd <- odd[, .(src_subject_id, odd_symptom_count)]
  } else {
    odd <- NULL
  }
}

# --- Merge everything ---------------------------------------------------------

merged <- classes
for (dt in list(cbcls, ksad, pgbi, dmdd, odd)) {
  if (!is.null(dt)) merged <- merge(merged, dt, by = "src_subject_id", all.x = TRUE)
}

# --- Count non-missing subjects per variable per class ------------------------

# These are the 12 plotted variables — mapped to actual column names
plot_vars <- c(
  "Total Problems"         = "cbcl_scr_syn_totprob_t",
  "Externalizing"          = "cbcl_scr_syn_external_t",
  "Internalizing"          = "cbcl_scr_syn_internal_t",
  "Aggressive Behavior"    = "cbcl_scr_syn_aggressive_t",
  "Attention Problems"     = "cbcl_scr_syn_attention_t",
  "ADHD (DSM5)"            = "cbcl_scr_dsm5_adhd_t",
  "ODD (DSM5)"             = "cbcl_scr_dsm5_opposit_t",
  "Depression (DSM5)"      = "cbcl_scr_dsm5_depress_t",
  "Irritability (K-SADS)"  = "irritability_score",
  "Emotion Dysreg (PGBI)"  = "pgbi_total",
  "DMDD Symptoms"          = "dmdd_symptom_count",
  "ODD Symptoms"           = "odd_symptom_count"
)

# Keep only variables that were actually loaded
plot_vars <- plot_vars[plot_vars %in% names(merged)]

count_df <- map_dfr(names(plot_vars), function(label) {
  var <- plot_vars[[label]]
  merged %>%
    as_tibble() %>%
    group_by(latent_class) %>%
    summarise(
      n_plotted  = sum(!is.na(.data[[var]])),
      n_missing  = sum( is.na(.data[[var]])),
      pct_missing = round(100 * mean(is.na(.data[[var]])), 1),
      .groups = "drop"
    ) %>%
    mutate(variable = label, column = var) %>%
    relocate(variable, column)
})

# Also add a total-across-classes row per variable
totals <- count_df %>%
  group_by(variable, column) %>%
  summarise(
    latent_class = factor("TOTAL"),
    n_plotted    = sum(n_plotted),
    n_missing    = sum(n_missing),
    pct_missing  = round(100 * n_missing / (n_plotted + n_missing), 1),
    .groups = "drop"
  )

count_df_full <- bind_rows(count_df, totals) %>%
  arrange(variable, latent_class)

# --- Print & save -------------------------------------------------------------

cat("\n===== SUBJECTS PLOTTED PER VARIABLE PER CLASS =====\n\n")
count_df_full %>%
  select(variable, latent_class, n_plotted, n_missing, pct_missing) %>%
  print(n = Inf)

out_path <- file.path(OUTPUT_DIR, "subjects_per_plot.csv")
write_csv(count_df_full, out_path)
cat(sprintf("\nSaved to: %s\n", out_path))