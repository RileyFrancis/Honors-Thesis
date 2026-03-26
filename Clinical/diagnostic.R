library(data.table)
ABCD_PATH <- "/shared/healthinfolab/datasets/ABCD/Package_1215452"

# Check DERS with skip=1 instead of skip=2
col_names <- names(fread(file.path(ABCD_PATH, "diff_emotion_reg_p01.txt"), nrows = 0))
ders <- fread(file.path(ABCD_PATH, "diff_emotion_reg_p01.txt"), skip = 1, col.names = col_names)
cat("DERS irritation values with skip=1:\n")
print(table(ders$ders_upset_irritation_p, useNA = "always"))

# Check which file actually has CBCL T-scores
for (f in c("abcd_cbcl01.txt", "abcd_cbcls01.txt", "abcd_bpm01.txt")) {
  cols <- names(fread(file.path(ABCD_PATH, f), nrows = 0))
  syn_cols <- grep("cbcl_scr_syn", cols, value = TRUE)
  cat(sprintf("\n%s — cbcl_scr_syn cols: %d found\n", f, length(syn_cols)))
  if (length(syn_cols) > 0) cat(paste(syn_cols[1:min(3,length(syn_cols))], collapse=", "), "\n")
}