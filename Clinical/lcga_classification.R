# ==============================================================================
# LCGA Class Characterization
# ==============================================================================

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

ABCD_PATH  <- Sys.getenv("ABCD_PATH",
                          unset = "/shared/healthinfolab/datasets/ABCD/Package_1215452")
CLASS_FILE <- Sys.getenv("LCGA_CLASS_FILE",
                          unset = file.path(getwd(), "lcga_output",
                                            "lcga_class_assignments.csv"))
OUTPUT_DIR <- Sys.getenv("LCGA_OUTPUT_DIR",
                          unset = file.path(getwd(), "lcga_output"))
BASELINE   <- "baseline_year_1_arm_1"

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

# --- 4. Helper ----------------------------------------------------------------

load_abcd <- function(filename, abcd_path, eventname = BASELINE, cols = NULL) {
  fpath <- file.path(abcd_path, filename)
  if (!file.exists(fpath)) { warning(sprintf("File not found: %s", fpath)); return(NULL) }
  col_names <- names(fread(fpath, nrows = 0))
  row2_val  <- as.character(fread(fpath, skip = 1, nrows = 1, header = FALSE)[[1]])
  n_skip    <- if (row2_val == col_names[1]) 2L else 1L
  dt <- fread(fpath, skip = n_skip, col.names = col_names,
              na.strings = c("", "NA", "999", "777", "888", "999.0"))
  if (!is.null(eventname) && "eventname" %in% names(dt)) {
    ev <- eventname; dt <- dt[dt$eventname == ev]
  }
  id_cols <- intersect(c("src_subject_id", "eventname"), names(dt))
  if (!is.null(cols)) { keep <- union(id_cols, intersect(cols, names(dt))); dt <- dt[, ..keep] }
  cat(sprintf("  Loaded %-30s  %d rows, %d cols\n", filename, nrow(dt), ncol(dt)))
  dt
}

# --- 5. Load Data -------------------------------------------------------------

cat("\nLoading ABCD data files...\n")

dem <- load_abcd("pdem02.txt", ABCD_PATH)
if (!is.null(dem)) {
  dem[, age_years := as.numeric(interview_age) / 12]
  dem[, sex := factor(fcase(
    as.integer(demo_sex_v2) == 1, "Male", as.integer(demo_sex_v2) == 2, "Female",
    as.integer(demo_sex_v2) == 3, "Intersex-Male", as.integer(demo_sex_v2) == 4, "Intersex-Female",
    default = NA_character_), levels = c("Male", "Female", "Intersex-Male", "Intersex-Female"))]
  to_int <- function(x) suppressWarnings(as.integer(x))
  dem[, race_eth := fcase(
    to_int(demo_ethn_v2) == 1, "Hispanic", to_int(demo_race_a_p___11) == 1, "Black",
    to_int(demo_race_a_p___18)==1|to_int(demo_race_a_p___19)==1|to_int(demo_race_a_p___20)==1|
      to_int(demo_race_a_p___21)==1|to_int(demo_race_a_p___22)==1|to_int(demo_race_a_p___23)==1|
      to_int(demo_race_a_p___24)==1, "Asian",
    to_int(demo_race_a_p___10) == 1, "White", default = "Other/Multiracial")]
  dem[, race_eth    := factor(race_eth, levels = c("White","Black","Hispanic","Asian","Other/Multiracial"))]
  dem[, income_group := fcase(
    as.integer(demo_comb_income_v2) %in% 1:6,  "< $50k",
    as.integer(demo_comb_income_v2) %in% 7:8,  "$50k-$99k",
    as.integer(demo_comb_income_v2) %in% 9:10, ">= $100k")]
  dem[, income_group := factor(income_group, levels = c("< $50k","$50k-$99k",">= $100k"))]
  dem[, parent_edu  := fcase(
    as.integer(demo_prnt_ed_v2) %in% 0:12,  "No college",
    as.integer(demo_prnt_ed_v2) %in% 13:14, "Some college",
    as.integer(demo_prnt_ed_v2) %in% 15:21, "College+")]
  dem[, parent_edu  := factor(parent_edu, levels = c("No college","Some college","College+"))]
  dem[, married := factor(fifelse(as.integer(demo_prnt_marital_v2)==1,"Married","Not married"),
                          levels = c("Married","Not married"))]
  dem <- dem[, .(src_subject_id, age_years, sex, race_eth, income_group, parent_edu, married)]
}

cbcl_cols <- c(
  "cbcl_scr_syn_anxdep_t","cbcl_scr_syn_withdep_t","cbcl_scr_syn_somatic_t",
  "cbcl_scr_syn_social_t","cbcl_scr_syn_thought_t","cbcl_scr_syn_attention_t",
  "cbcl_scr_syn_rulebreak_t","cbcl_scr_syn_aggressive_t",
  "cbcl_scr_syn_internal_t","cbcl_scr_syn_external_t","cbcl_scr_syn_totprob_t",
  "cbcl_scr_dsm5_adhd_t","cbcl_scr_dsm5_depress_t","cbcl_scr_dsm5_anxdisord_t",
  "cbcl_scr_dsm5_opposit_t","cbcl_scr_dsm5_conduct_t")
cbcl  <- load_abcd("abcd_cbcls01.txt", ABCD_PATH, cols = cbcl_cols)
cbcls <- NULL

ksad <- load_abcd("abcd_ksad01.txt", ABCD_PATH)
if (!is.null(ksad)) {
  clamp_binary <- function(x) { x <- suppressWarnings(as.numeric(x)); x[!x %in% c(0,1)] <- NA; x }
  adhd_items   <- grep("^ksads_14_", names(ksad), value = TRUE)
  if (length(adhd_items) > 0) {
    ksad[, (adhd_items) := lapply(.SD, clamp_binary), .SDcols = adhd_items]
    ksad[, adhd_symptom_count := rowSums(.SD, na.rm = TRUE), .SDcols = adhd_items]
  } else { ksad[, adhd_symptom_count := NA_real_] }
  irr_items   <- c("ksads_1_3_p","ksads_3_229_p","ksads_15_432_p","ksads_15_91_p")
  irr_present <- intersect(irr_items, names(ksad))
  if (length(irr_present) > 0) {
    ksad[, (irr_present) := lapply(.SD, clamp_binary), .SDcols = irr_present]
    ksad[, ksads_irritability_sum := rowSums(.SD, na.rm = TRUE), .SDcols = irr_present]
  } else { ksad[, ksads_irritability_sum := NA_real_] }
  ksad <- ksad[, .(src_subject_id, adhd_symptom_count, ksads_irritability_sum)]
}

ders_raw <- load_abcd("diff_emotion_reg_p01.txt", ABCD_PATH, eventname = NULL)
ders <- NULL
if (!is.null(ders_raw) && nrow(ders_raw) > 0) {
  upset_items <- grep("^ders_upset_", names(ders_raw), value = TRUE)
  if (length(upset_items) > 0) {
    clamp15 <- function(x) { x <- suppressWarnings(as.numeric(x)); x[x < 1 | x > 5] <- NA; x }
    ders_raw[, (upset_items) := lapply(.SD, clamp15), .SDcols = upset_items]
    ders_raw[, ders_total     := rowSums(.SD, na.rm = TRUE), .SDcols = upset_items]
    ders_raw[, ders_irritation := if ("ders_upset_irritation_p" %in% names(ders_raw))
                                    clamp15(ders_upset_irritation_p) else NA_real_]
    ders_raw[, interview_age := suppressWarnings(as.numeric(interview_age))]
    setorder(ders_raw, src_subject_id, interview_age)
    ders <- unique(ders_raw[, .(src_subject_id, ders_total, ders_irritation)], by = "src_subject_id")
    cat(sprintf("  DERS: %d subjects (earliest timepoint)\n", nrow(ders)))
  }
}

odd_raw <- load_abcd("opp_defiant_disorder_p01.txt", ABCD_PATH)
odd <- NULL
if (!is.null(odd_raw) && nrow(odd_raw) > 0) {
  odd_candidates <- grep("^ksads_odd_raw_", names(odd_raw), value = TRUE)
  is_binary <- function(col) { vals <- suppressWarnings(as.numeric(odd_raw[[col]])); all(is.na(vals)|vals %in% c(0,1)) }
  odd_items <- Filter(is_binary, odd_candidates)
  cat(sprintf("  ODD: %d binary items retained out of %d\n", length(odd_items), length(odd_candidates)))
  if (length(odd_items) > 0) {
    odd_raw[, (odd_items) := lapply(.SD, function(x) suppressWarnings(as.numeric(x))), .SDcols = odd_items]
    odd_raw[, odd_symptom_count := rowSums(.SD, na.rm = TRUE), .SDcols = odd_items]
    odd <- odd_raw[, .(src_subject_id, odd_symptom_count)]
  }
}

# --- 6. Merge -----------------------------------------------------------------

cat("\nMerging data...\n")
merged <- classes
for (dt in list(dem, cbcl, cbcls, ksad, ders, odd)) {
  if (!is.null(dt)) merged <- merge(merged, dt, by = "src_subject_id", all.x = TRUE)
}
cat(sprintf("Final merged dataset: %d subjects, %d variables\n", nrow(merged), ncol(merged)))

# --- 7. Variable Lists --------------------------------------------------------

continuous_vars <- c(
  "age_years",
  "cbcl_scr_syn_anxdep_t","cbcl_scr_syn_withdep_t","cbcl_scr_syn_somatic_t",
  "cbcl_scr_syn_social_t","cbcl_scr_syn_thought_t","cbcl_scr_syn_attention_t",
  "cbcl_scr_syn_rulebreak_t","cbcl_scr_syn_aggressive_t",
  "cbcl_scr_syn_internal_t","cbcl_scr_syn_external_t","cbcl_scr_syn_totprob_t",
  "cbcl_scr_dsm5_adhd_t","cbcl_scr_dsm5_depress_t","cbcl_scr_dsm5_anxdisord_t",
  "cbcl_scr_dsm5_opposit_t","cbcl_scr_dsm5_conduct_t",
  "adhd_symptom_count","ksads_irritability_sum","ders_total","ders_irritation","odd_symptom_count")
categorical_vars <- c("sex","race_eth","income_group","parent_edu","married")

has_data <- function(var) var %in% names(merged) && sum(!is.na(merged[[var]])) > 0
continuous_vars  <- Filter(has_data, continuous_vars)
categorical_vars <- Filter(has_data, categorical_vars)
cat(sprintf("\nContinuous: %d  |  Categorical: %d\n", length(continuous_vars), length(categorical_vars)))

# --- 8. Statistical Tests -----------------------------------------------------

cat("\nRunning Kruskal-Wallis tests...\n")
kw_results <- lapply(continuous_vars, function(var) {
  df <- merged[!is.na(get(var)), .(value = get(var), latent_class)]
  if (length(unique(df$latent_class)) < 2) return(NULL)
  kw   <- kruskal.test(value ~ latent_class, data = df)
  desc <- df[, .(mean = round(mean(value,na.rm=TRUE),2), sd = round(sd(value,na.rm=TRUE),2),
                 n = sum(!is.na(value))), by = latent_class]
  list(var = var, kw_p = kw$p.value, desc = desc)
})
kw_results <- Filter(Negate(is.null), kw_results)

cat("Running Dunn pairwise tests...\n")
dunn_results <- lapply(kw_results, function(res) {
  df <- merged[!is.na(get(res$var)), .(value = get(res$var), latent_class)]
  if (length(unique(df$latent_class)) < 2) return(NULL)
  out <- dunn.test(df$value, df$latent_class, method="none", kw=FALSE,
                   label=TRUE, wrap=FALSE, table=FALSE, list=FALSE, rmc=FALSE, alpha=0.05, altp=FALSE)
  tibble(variable = res$var, comparison = out$comparisons, Z = round(out$Z,3), p_raw = out$P)
})
dunn_df       <- bind_rows(dunn_results)
dunn_df$p_fdr <- p.adjust(dunn_df$p_raw, method = "fdr")
dunn_df$sig   <- case_when(dunn_df$p_fdr<0.001~"***", dunn_df$p_fdr<0.01~"**",
                           dunn_df$p_fdr<0.05~"*", TRUE~"ns")

cat("Running chi-square tests...\n")
chi_results <- lapply(categorical_vars, function(var) {
  tab <- table(merged[[var]], merged$latent_class, useNA = "no")
  chi <- tryCatch(chisq.test(tab), error = function(e) NULL)
  list(var = var, chi_p = if (!is.null(chi)) chi$p.value else NA,
       table = as.data.frame(prop.table(tab, margin = 2) * 100))
})
all_p   <- c(sapply(kw_results, `[[`, "kw_p"), sapply(chi_results, `[[`, "chi_p"))
all_fdr <- p.adjust(all_p, method = "fdr")

# --- 9. Summary Table ---------------------------------------------------------

cat("\nBuilding summary table...\n")
n_cont    <- length(continuous_vars)
cont_rows <- lapply(seq_along(kw_results), function(i) {
  res <- kw_results[[i]]; desc <- res$desc
  row <- tibble(Variable = res$var, Type = "Continuous")
  for (cls in levels(merged$latent_class)) {
    d <- desc[latent_class == cls]
    row[[cls]] <- if (nrow(d) > 0) sprintf("%.2f (%.2f)", d$mean, d$sd) else "—"
  }
  row$p_fdr <- round(all_fdr[i], 4)
  row$sig   <- case_when(row$p_fdr<0.001~"***", row$p_fdr<0.01~"**", row$p_fdr<0.05~"*", TRUE~"ns")
  row
})
cat_rows <- lapply(seq_along(chi_results), function(i) {
  res <- chi_results[[i]]
  tab <- as.data.table(res$table); setnames(tab, c("level","latent_class","pct"))
  counts <- as.data.table(table(merged[[res$var]], merged$latent_class, useNA="no"))
  setnames(counts, c("level","latent_class","n"))
  tab <- merge(tab, counts, by = c("level","latent_class"))
  lapply(unique(tab$level), function(lvl) {
    row <- tibble(Variable = sprintf("%s: %s", res$var, lvl), Type = "Categorical")
    for (cls in levels(merged$latent_class)) {
      d <- tab[level == lvl & latent_class == cls]
      row[[cls]] <- if (nrow(d) > 0) sprintf("%d (%.1f%%)", d$n, d$pct) else "—"
    }
    row$p_fdr <- round(all_fdr[n_cont + i], 4)
    row$sig   <- case_when(row$p_fdr<0.001~"***", row$p_fdr<0.01~"**", row$p_fdr<0.05~"*", TRUE~"ns")
    row
  })
})
summary_table <- bind_rows(c(cont_rows, unlist(cat_rows, recursive = FALSE)))
cat("\n===== CHARACTERIZATION SUMMARY =====\n")
print(summary_table, n = Inf)

# --- 10. Visualization --------------------------------------------------------
cat("\nGenerating plots...\n")

# Single shared palette — same hue for both violin fills and bar fills.
# Violins use alpha=0.6 so they naturally appear lighter; bars use alpha=1
# (full saturation). This ensures consistent identity across plot types.
class_palette <- c(
  "Class 1\nLow-Stable"      = "#4f8bc8",
  "Class 2\nHigh-Decreasing" = "#8b3fca",
  "Class 3\nLow-Increasing"  = "#cb6587"
)

# CBCL T-score variables — always rendered as violins
cbcl_t_vars <- c(
  "cbcl_scr_syn_totprob_t","cbcl_scr_syn_external_t","cbcl_scr_syn_internal_t",
  "cbcl_scr_syn_aggressive_t","cbcl_scr_syn_attention_t","cbcl_scr_dsm5_adhd_t",
  "cbcl_scr_dsm5_opposit_t","cbcl_scr_dsm5_depress_t")

# Classify each plot variable as continuous (violin) or discrete (bar chart).
# CBCL T-scores are always violin even though they are integers.
# odd_symptom_count is always bar even though it has >15 unique values
# (zero-inflated counts look poor as violins).
# Everything else: discrete = integer-valued with <=15 unique values.
always_violin <- cbcl_t_vars
always_bar    <- "odd_symptom_count"

is_discrete_var <- function(var) {
  if (var %in% always_violin) return(FALSE)
  if (var %in% always_bar)    return(TRUE)
  vals <- merged[[var]][!is.na(merged[[var]])]
  all(vals == floor(vals)) && length(unique(vals)) <= 15
}

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

y_axis_label <- function(var) {
  case_when(
    var %in% cbcl_t_vars            ~ "T-score (mean=50, SD=10)",
    var == "ksads_irritability_sum" ~ "Items endorsed (0=none, 4=all)",
    var == "ders_total"             ~ "Sum score (20 items, 1\u20135 scale)",
    var == "ders_irritation"        ~ "Rating (1=Never, 5=Always)",
    var == "odd_symptom_count"      ~ "Symptoms endorsed (out of 25)",
    TRUE                            ~ "Value"
  )
}

plot_vars <- intersect(
  c(cbcl_t_vars, "ksads_irritability_sum", "ders_total", "ders_irritation", "odd_symptom_count"),
  names(merged)
)

# Shared theme base — keeps font sizes and panel style identical across types
base_theme <- theme_bw(base_size = 11) +
  theme(legend.position = "none",       # legends removed; shared legend added below
        axis.text.x     = element_text(size = 8),
        plot.title      = element_text(size = 10, face = "bold"))

make_violin <- function(var) {
  df_plot <- as.data.frame(merged[!is.na(get(var)), .(latent_class, value = get(var))])
  if (nrow(df_plot) < 10 || length(unique(df_plot$latent_class)) < 2) return(NULL)
  label <- ifelse(var %in% names(var_labels), var_labels[[var]], var)

  ggplot(df_plot, aes(x = latent_class, y = value, fill = latent_class)) +
    geom_violin(trim = TRUE, alpha = 0.55, color = NA, scale = "width") +
    geom_boxplot(width = 0.15, outlier.size = 0.3, fill = "white", color = "grey30") +
    scale_fill_manual(values = class_palette) +
    labs(title = label, x = NULL, y = y_axis_label(var)) +
    base_theme
}

make_bar <- function(var) {
  df_plot <- as.data.frame(merged[!is.na(get(var)), .(latent_class, value = get(var))])
  if (nrow(df_plot) < 10 || length(unique(df_plot$latent_class)) < 2) return(NULL)
  label <- ifelse(var %in% names(var_labels), var_labels[[var]], var)

  df_pct <- df_plot %>%
    mutate(value = factor(value)) %>%
    count(latent_class, value) %>%
    group_by(latent_class) %>%
    mutate(pct = 100 * n / sum(n)) %>%
    ungroup()

  # alpha = 1 so bars are fully saturated — same hue as violins, just no transparency
  p <- ggplot(df_pct, aes(x = value, y = pct, fill = latent_class)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7, alpha = 1) +
    scale_fill_manual(values = class_palette) +
    labs(title = label, x = y_axis_label(var), y = "% of class") +
    base_theme

  # Log-scale y-axis for zero-inflated count variables (e.g. ODD symptoms)
  # Uses log10(x + 1) pseudo-log so zero bars are still visible
  if (var == "odd_symptom_count") {
    p <- p +
      scale_y_continuous(
        trans  = "log1p",
        breaks = c(0, 1, 2, 5, 10, 25, 50, 75),
        labels = c("0", "1", "2", "5", "10", "25", "50", "75")
      ) +
      labs(y = "% of class (log scale)")
  }
  p
}

all_plots <- lapply(plot_vars, function(var) {
  if (!var %in% names(merged)) return(NULL)
  if (is_discrete_var(var)) make_bar(var) else make_violin(var)
})
all_plots <- Filter(Negate(is.null), all_plots)

n_plots <- length(all_plots)
n_cols  <- 4L
n_rows  <- ceiling(n_plots / n_cols)

# Build a standalone legend panel using a dummy plot
legend_data <- data.frame(
  latent_class = factor(names(class_palette), levels = names(class_palette)),
  x = 1, y = 1
)
legend_plot <- ggplot(legend_data, aes(x = x, y = y, fill = latent_class)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(
    values = class_palette,
    name   = NULL,
    guide  = guide_legend(direction = "horizontal", nrow = 1,
                          keywidth = unit(1.2, "cm"), keyheight = unit(0.5, "cm"),
                          label.theme = element_text(size = 11))
  ) +
  theme_void() +
  theme(legend.position = "bottom",
        legend.margin   = margin(0, 0, 0, 0))

# Extract just the legend grob
library(ggplot2)
get_legend_grob <- function(p) {
  g <- ggplotGrob(p)
  leg_idx <- which(sapply(g$grobs, function(x) x$name) == "guide-box")
  if (length(leg_idx) == 0) return(NULL)
  g$grobs[[leg_idx]]
}
legend_grob <- get_legend_grob(legend_plot)

# Arrange plots + legend: plots fill the top, legend sits in a thin strip below
plots_grob <- gridExtra::arrangeGrob(grobs = all_plots, ncol = n_cols, nrow = n_rows)

if (!is.null(legend_grob)) {
  final_grob <- gridExtra::arrangeGrob(
    plots_grob,
    legend_grob,
    nrow   = 2,
    heights = unit(c(4.5 * n_rows, 0.5), c("in", "in"))
  )
} else {
  final_grob <- plots_grob
}

ggsave(file.path(OUTPUT_DIR, "characterization_cbcl_plots.png"),
       final_grob,
       width  = 16,
       height = 4.5 * n_rows + 0.5,
       dpi    = 150)

# Demographic bar charts — also no per-plot legend; shared legend appended below
demo_plots <- lapply(categorical_vars, function(var) {
  df_plot <- as.data.frame(merged[!is.na(get(var)), .(latent_class, value = get(var))])
  df_pct  <- df_plot %>%
    count(latent_class, value) %>%
    group_by(latent_class) %>%
    mutate(pct = 100 * n / sum(n)) %>%
    ungroup()
  ggplot(df_pct, aes(x = value, y = pct, fill = latent_class)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 1) +
    scale_fill_manual(values = class_palette) +
    labs(title = var, x = NULL, y = "% within class") +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 30, hjust = 1, size = 9))
})

n_demo_cols  <- min(3L, length(demo_plots))
demo_grob    <- gridExtra::arrangeGrob(
  grobs = demo_plots,
  ncol  = n_demo_cols,
  nrow  = ceiling(length(demo_plots) / n_demo_cols)
)

if (!is.null(legend_grob)) {
  demo_final <- gridExtra::arrangeGrob(
    demo_grob,
    legend_grob,
    nrow    = 2,
    heights = unit(c(5 * ceiling(length(demo_plots) / n_demo_cols), 0.5), c("in", "in"))
  )
} else {
  demo_final <- demo_grob
}

ggsave(file.path(OUTPUT_DIR, "characterization_demo_plots.png"),
       demo_final,
       width  = 14,
       height = 5 * ceiling(length(demo_plots) / n_demo_cols) + 0.5,
       dpi    = 150)

cat("Plots saved.\n")

# --- 11. Save Results ---------------------------------------------------------

fwrite(summary_table, file.path(OUTPUT_DIR, "characterization_summary.csv"))
fwrite(dunn_df,       file.path(OUTPUT_DIR, "characterization_pairwise.csv"))

cat("\nSaved:\n")
cat(sprintf("  %s/characterization_summary.csv\n",    OUTPUT_DIR))
cat(sprintf("  %s/characterization_pairwise.csv\n",   OUTPUT_DIR))
cat(sprintf("  %s/characterization_cbcl_plots.png\n", OUTPUT_DIR))
cat(sprintf("  %s/characterization_demo_plots.png\n", OUTPUT_DIR))
cat("\n===== CHARACTERIZATION COMPLETE =====\n")