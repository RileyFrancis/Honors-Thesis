# ==============================================================================
# Targeted diagnostic for Emotion Dysregulation and K-SADS Irritability
# ==============================================================================

library(data.table)

ABCD_PATH <- Sys.getenv("ABCD_PATH",
                         unset = "/shared/healthinfolab/datasets/ABCD/Package_1215452")
BASELINE  <- "baseline_year_1_arm_1"

peek <- function(filename, n_rows = 5) {
  fpath <- file.path(ABCD_PATH, filename)
  if (!file.exists(fpath)) { cat(sprintf("  NOT FOUND: %s\n", fpath)); return(NULL) }
  col_names <- names(fread(fpath, nrows = 0))
  dt <- fread(fpath, skip = 2, col.names = col_names,
              na.strings = c("", "NA", "999", "777", "999.0"))
  if ("eventname" %in% names(dt)) dt <- dt[eventname == BASELINE]
  cat(sprintf("\n%s  —  %d rows, %d cols\n", filename, nrow(dt), ncol(dt)))
  cat(sprintf("  All columns:\n    %s\n", paste(col_names, collapse="\n    ")))
  # Show first few non-ID rows for all columns
  cat(sprintf("  First %d rows:\n", n_rows))
  print(head(dt, n_rows))
  dt
}

# --- 1. Emotion dysregulation candidates --------------------------------------
cat("========== EMOTION DYSREGULATION FILES ==========\n")

# Check the follow-up version first
peek("soc_dev_fu_diff_emo_reg_p01.txt")

# Check the original diff_emotion_reg_p01
peek("diff_emotion_reg_p01.txt")

# abcd_mhy02 — mental health yearly
mhy <- peek("abcd_mhy02.txt")
if (!is.null(mhy)) {
  emo_cols <- grep("emo|dysreg|reg|ders|irrit|anger|temper",
                   names(mhy), value = TRUE, ignore.case = TRUE)
  cat(sprintf("  Emotion-related cols: %s\n", paste(emo_cols, collapse = ", ")))
}

# --- 2. K-SADS irritability ---------------------------------------------------
cat("\n========== K-SADS IRRITABILITY ITEMS ==========\n")

ksad_fpath <- file.path(ABCD_PATH, "abcd_ksad01.txt")
col_names  <- names(fread(ksad_fpath, nrows = 0))
dt_ksad    <- fread(ksad_fpath, skip = 2, col.names = col_names,
                    na.strings = c("", "NA", "999", "777", "999.0"))
dt_ksad    <- dt_ksad[eventname == BASELINE]

# Check the ksads_1_3 and ksads_1_4 items (DMDD irritability in ABCD R5)
dmdd_candidates <- c("ksads_1_3_p", "ksads_1_4_p",
                     "ksads_1_840_p", "ksads_1_841_p", "ksads_1_843_p",
                     "ksads_1_156_p", "ksads_1_163_p", "ksads_1_164_p")
present <- intersect(dmdd_candidates, names(dt_ksad))
cat(sprintf("Candidate irritability columns present: %s\n",
            paste(present, collapse = ", ")))

if (length(present) > 0) {
  cat("\nValue distributions:\n")
  for (col in present) {
    vals <- dt_ksad[[col]]
    cat(sprintf("  %-25s  n_nonNA=%d  table: ", col, sum(!is.na(vals))))
    print(table(vals, useNA = "always"))
  }
}

# Also check disruptive_mood_dysreg_p01
cat("\n--- disruptive_mood_dysreg_p01.txt ---\n")
dmdd_dt <- peek("disruptive_mood_dysreg_p01.txt", n_rows = 3)

# --- 3. PGBI (General Behavior Inventory — mood dysregulation) ---------------
cat("\n--- abcd_pgbi01.txt (General Behavior Inventory) ---\n")
peek("abcd_pgbi01.txt", n_rows = 3)