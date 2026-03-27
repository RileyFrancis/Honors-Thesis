library(data.table)
ABCD_PATH <- "/shared/healthinfolab/datasets/ABCD/Package_1215452"

col_names <- names(fread(file.path(ABCD_PATH, "abcd_ksad01.txt"), nrows = 0))
row2_val  <- as.character(fread(file.path(ABCD_PATH, "abcd_ksad01.txt"),
                                 skip = 1, nrows = 1, header = FALSE)[[1]])
n_skip <- if (row2_val == col_names[1]) 2L else 1L

ksad <- fread(file.path(ABCD_PATH, "abcd_ksad01.txt"), skip = n_skip,
              col.names = col_names,
              na.strings = c("", "NA", "999", "777", "999.0"))

ev <- "baseline_year_1_arm_1"
ksad <- ksad[ksad$eventname == ev]

cat("Rows at baseline:", nrow(ksad), "\n")
for (col in c("ksads_1_3_p", "ksads_3_226_p", "ksads_15_432_p", "ksads_15_91_p")) {
  cat(sprintf("\n%s:\n", col))
  print(table(ksad[[col]], useNA = "always"))
}
