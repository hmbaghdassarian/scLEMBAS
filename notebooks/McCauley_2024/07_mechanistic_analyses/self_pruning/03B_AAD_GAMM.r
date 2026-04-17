library(mgcv)
library(ggplot2)
# library(gridExtra)

library(future.apply)
library(future)
library(progressr)

data_path = '/home/hmbaghda/orcd/pool/scLEMBAS/analysis'
author = 'McCauley'
n_workers<-1
plan(multisession, workers = n_workers)
seed = 888

res<-read.csv(file.path(data_path, 'processed', 'pruning_AAD.csv'), row.names = 1)

head(res)

res$log_MAE <- log(1e-20 + res$MAE)
res$log_connectivity <- log(res$connectivity)
res$log_n_edges <- log(res$n_edges)
res$log_mean_weight <- log(res$mean_weight)
res$type <- as.factor(res$type)
res$model_id <- as.factor(res$model_id)

res$n_edges_resid <- residuals(lm(log_n_edges ~ log_connectivity, data=res)) # due to high concurvity / multi-collinearity of edges with connectivity



k_effect = 15

model_full <- gam(log_MAE ~ 
                   s(log_mean_weight, k=k_effect) +    # Weight effect
                   s(log_connectivity, k=k_effect) +        # Topology effect  
                   s(model_id, bs="re") +        # Random intercepts
                  s(n_edges_resid, k=k_effect) +              # Control for no. of zeroed edges
                   type,                         # Fixed effect
                 data=res, 
#                   family = Gamma(link="log"),
                 method="REML")

summary_full <- summary(model_full)
summary_full


par(mfrow=c(2,2))
gam.check(model_full)

cat("\nChecking for concurvity (similar to multicollinearity):\n")
print(concurvity(model_full, full=FALSE))
cat("\nValues close to 1 indicate problematic concurvity.\n")
cat("Values < 0.8 are generally acceptable.\n")

# fit with ML for comparison
model_full_ml <- gam(log_MAE ~ 
                   s(log_mean_weight, k=k_effect) +
                   s(log_connectivity, k=k_effect) +      
                   s(model_id, bs="re") +      
                   s(n_edges_resid, k=k_effect) +           
                   type,
                 data = res,
                 method = "ML")

model_reduced_ml <- gam(log_MAE ~ 
                   s(log_connectivity, k=k_effect) +      
                   s(model_id, bs="re") +      
                   s(n_edges_resid, k=k_effect) +           
                   type,
                 data = res,
                 method = "ML")


# AIC comparison
aic_res <- AIC(model_reduced_ml, model_full_ml)

aic_reduced <- aic_res["model_reduced_ml", "AIC"]
aic_full    <- aic_res["model_full_ml", "AIC"]
delta_aic   <- aic_reduced - aic_full   # positive => full better

print(aic_res)

cat(sprintf(
  "\nAIC comparison:\nReduced model AIC = %.2f\nFull model AIC = %.2f\nDelta AIC = %.2f\n",
  aic_reduced, aic_full, delta_aic
))

if (delta_aic < 2) {
  interp <- "Models have equivalent support."
} else if (delta_aic < 7) {
  interp <- "Weight provides modest additional explanatory value."
} else if (delta_aic < 10) {
  interp <- "Weight clearly improves model fit."
} else {
  interp <- "Weight strongly improves model fit, indicating an independent contribution beyond topology and controls."
}

cat(sprintf("Interpretation: %s\n", interp))


plot_gam_effects_1x3 <- function(
  model_full,
  w = 10, h = 3.5,
  base_cex = 1.3,
  axis_cex = 1.2,
  lab_cex  = 1.4,
  main_cex = 1.2,
  lwd = 2,
  save_path = NULL,
  dpi = 600
) {
  blue_est  <- "#1f77b4"
  gray_zero <- "gray60"

  if (!is.null(save_path)) {
    grDevices::png(
      filename = save_path,
      width = w, height = h, units = "in",
      res = dpi, type = "cairo", bg = "transparent"
    )
    on.exit(grDevices::dev.off(), add = TRUE)
  }

  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)

  par(
    mfrow = c(1, 3),
    mar = c(4.6, 4.8, 1.5, 0.8),
    cex = base_cex,
    cex.axis = axis_cex,
    cex.lab  = lab_cex,
    cex.main = main_cex
  )

  # 1. log mean weight
  mgcv::plot.gam(
    model_full, select = 1,
    se = TRUE,
    col = blue_est, lwd = lwd,
    xlab = "log(Mean |Edge Weight|)",
    ylab = "log(MAE)",
    rug = FALSE,
    residuals = FALSE
  )
  abline(h = 0, col = gray_zero, lwd = 1)

  # 2. log connectivity
  mgcv::plot.gam(
    model_full, select = 2,
    se = TRUE,
    col = blue_est, lwd = lwd,
    xlab = "log(Edge Connectivity)",
    ylab = "log(MAE)",
    rug = FALSE,
    residuals = FALSE
  )
  abline(h = 0, col = gray_zero, lwd = 1)

  # 3. residualized number of removed edges
  mgcv::plot.gam(
    model_full, select = 4,
    se = TRUE,
    main = "",
    col = blue_est, lwd = lwd,
    xlab = "Residualized log(No. of Removed Edges)",
    ylab = "log(MAE)",
    rug = FALSE,
    residuals = FALSE
  )
  abline(h = 0, col = gray_zero, lwd = 1)

  invisible(NULL)
}

plot_gam_effects_1x3 <- function(
  model_full,
  w = 10, h = 3.5,
  
  axis_cex = 1.2,
  label_cex = 1.6,
  base_cex = 1.2,
  
  mar_vec = c(6, 5.5, 1.5, 0.8),
  
  lwd = 2,
  save_path = NULL,
  dpi = 600
) {
  
  blue_est  <- "#1f77b4"
  gray_zero <- "gray60"

  if (!is.null(save_path)) {
    grDevices::png(
      filename = save_path,
      width = w, height = h, units = "in",
      res = dpi, type = "cairo", bg = "transparent"
    )
    on.exit(grDevices::dev.off(), add = TRUE)
  }

  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)

  par(
    mfrow = c(1, 3),
    mar = c(5.2, 4.2, 1.2, 0.3),
    cex = base_cex,
    cex.axis = axis_cex
  )

  ylab_txt <- "log(MAE)"

  # 👇 add x_line argument
  plot_panel <- function(select_idx, xlab_txt, x_line = 2.5) {
    mgcv::plot.gam(
      model_full, select = select_idx,
      se = TRUE,
      col = blue_est, lwd = lwd,
      xlab = "", ylab = "",
      rug = FALSE,
      residuals = FALSE
    )
    abline(h = 0, col = gray_zero, lwd = 1)
    
    mtext(xlab_txt, side = 1, line = x_line, cex = label_cex)
    mtext(ylab_txt, side = 2, line = 2.5, cex = label_cex)
  }

  # panels
  plot_panel(1, "log(Mean |Edge Weight|)")
  plot_panel(2, "log(Edge Connectivity)")
  plot_panel(4, "Residualized\nlog(No. Removed Edges)", x_line = 3.5)  # 👈 only change here

  invisible(NULL)
}

w = 20
h = 5
options(repr.plot.width = w, repr.plot.height = h)

# save
plot_gam_effects_1x3(
  model_full,
  w = w, h = h,
  base_cex = 1.4,
  axis_cex = 1.2,
  label_cex = 1.5,
  save_path = file.path(data_path,'figures', "McCauley_selfpruning_GAM_partial_effects.png"),
  dpi = 1200
)

# display
plot_gam_effects_1x3(
  model_full,
  w = w, h = h,
  base_cex = 1.4,
  axis_cex = 1.2,
  label_cex = 1.5,
  save_path = NULL
)

type_coef <- coef(model_full)["typetrue"]  # Assuming 'stochastic' is reference
type_se <- summary(model_full)$p.table["typetrue", "Std. Error"]
type_pval <- summary(model_full)$p.table["typetrue", "Pr(>|t|)"]

cat(sprintf("Coefficient: %.4f (SE = %.4f, p = %.4f)\n", type_coef, type_se, type_pval))
cat(sprintf("\nInterpretation: Zeroing TRUE edges changes AAD by %.4f units\n", type_coef))
cat(sprintf("compared to zeroing STOCHASTIC edges (holding all else constant).\n"))

if (type_coef > 0) {
  cat("Direction: Removing true edges INCREASES AAD relative to stochastic edges.\n")
} else {
  cat("Direction: Removing true edges DECREASES AAD relative to stochastic edges.\n")
}



model_full2 <- gam(log_MAE ~ 
                   type + # still a fixed effect
                   s(log_mean_weight, by = type, k=k_effect) +    # Weight effect
                   s(log_connectivity, k=k_effect) +        # Topology effect  
                   s(model_id, bs="re") +        # Random intercepts
                  s(n_edges_resid, k=k_effect),              # Control for no. of zeroed edges
                 data=res, 
#                   family = Gamma(link="log"),
                 method="REML")
summary(model_full2)

model_full_lrt <- gam(log_MAE ~ 
                   s(log_mean_weight, k=k_effect) +    # Weight effect
                   s(log_connectivity, k=k_effect) +        # Topology effect  
                   s(model_id, bs="re") +        # Random intercepts
                  s(n_edges_resid, k=k_effect) +              # Control for no. of zeroed edges
                   type,                         # Fixed effect
                 data=res, 
#                   family = Gamma(link="log"),
                 method="ML")

model_full2_lrt <- gam(log_MAE ~ 
                   type + # still a fixed effect
                   s(log_mean_weight, by = type, k=k_effect) +    # Weight effect
                   s(log_connectivity, k=k_effect) +        # Topology effect  
                   s(model_id, bs="re") +        # Random intercepts
                  s(n_edges_resid, k=k_effect),              # Control for no. of zeroed edges
                 data=res, 
#                   family = Gamma(link="log"),
                 method="ML")

# in this case, model_full_lrt is the reduced model relative to model_full2_lrt
anova(model_full_lrt, model_full2_lrt, test="Chisq")


options(repr.plot.width = 7, repr.plot.height = 5)

# grid over weight
w_grid <- seq(min(res$log_mean_weight),
              max(res$log_mean_weight),
              length.out = 200)

# helper
mode_level <- function(x) {
  tab <- table(x)
  names(tab)[which.max(tab)]
}

# baseline row
base <- res[1, , drop = FALSE]
base$log_connectivity <- median(res$log_connectivity, na.rm = TRUE)
base$n_edges_resid    <- median(res$n_edges_resid, na.rm = TRUE)

base$type     <- factor(mode_level(res$type), levels = levels(res$type))
base$model_id <- factor(mode_level(res$model_id), levels = levels(res$model_id))

# predictions by type
pred_by_type <- lapply(c("stochastic", "true"), function(tp) {
  nd <- base[rep(1, length(w_grid)), ]
  nd$log_mean_weight <- w_grid
  nd$type <- factor(tp, levels = levels(res$type))

  pr <- predict(model_full2,
                newdata = nd,
                se.fit = TRUE,
                exclude = "s(model_id)")

  data.frame(w = w_grid, fit = pr$fit, se = pr$se.fit, type = tp)
})

dfp <- do.call(rbind, pred_by_type)

# plot
plot(NULL,
     xlim = range(dfp$w),
     ylim = range(c(dfp$fit - 2*dfp$se, dfp$fit + 2*dfp$se)),
     xlab = "log(Mean |Edge Weight|)",
     ylab = "log(AAD)")

cols <- c(stochastic = "#ff7f0e", true = "#1f77b4")

for (tp in names(cols)) {
  d <- dfp[dfp$type == tp, ]
  lines(d$w, d$fit, col = cols[tp], lwd = 2)
  lines(d$w, d$fit + 2*d$se, col = cols[tp], lty = 2)
  lines(d$w, d$fit - 2*d$se, col = cols[tp], lty = 2)
}

abline(h = 0, col = "gray60", lwd = 2)

legend("topleft",
       legend = c("True Edges", "Stochastic Edges"),
       col = cols[c("true","stochastic")],
       lwd = 2,
       bty = "n")


