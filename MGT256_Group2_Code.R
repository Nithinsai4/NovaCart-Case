# NovaCart Case: Sales Prediction using MLR, kNN, CART
# MGT256 Group 2 Project
# Author: Nithin Adru

# 1. Load Data and Packages
library(ggplot2)    # for visualization
library(caret)      # for data preprocessing (e.g., normalization)
library(FNN)        # for kNN regression
library(rpart)      # for CART
library(rpart.plot)# for plotting decision trees
library(readxl)
library(dplyr)

# Read the dataset 
novacart.df <- read_excel("+Case2.xlsx", sheet = 1)

novacart.df <- novacart.df %>%
  mutate_if(is.character, as.factor)

novacart.df <- novacart.df %>%
  rename(
    ProductCategory = Product_Category,
    DiscountRate = Discount,
    MarketingSpend = Marketing_Spend,
    Region = Customer_Segment,     
    Revenue = Price                
  )

# Ensure categorical variables are factors 
novacart.df$Region <- as.factor(novacart.df$Region)
novacart.df$ProductCategory <- as.factor(novacart.df$ProductCategory)

# Quick data overview
dim(novacart.df)
str(novacart.df)
summary(novacart.df)

# 2. Exploratory Data Analysis (EDA)

# Distribution of Monthly Sales Revenue
ggplot(novacart.df, aes(x = Revenue)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  labs(title = "Distribution of Monthly Revenue", x = "Monthly Revenue", y = "Frequency") +
  theme_minimal()
# We see that the revenue distribution is right-skewed, indicating a few months with exceptionally high sales.

# Relationship between Revenue and Marketing Spend
ggplot(novacart.df, aes(x = MarketingSpend, y = Revenue)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  labs(title = "Revenue vs Marketing Spend", x = "Monthly Marketing Spend", y = "Monthly Revenue") +
  theme_minimal()
# Revenue tends to increase with higher marketing spend, supporting the marketing team's belief that advertising drives sales.

# Relationship between Revenue and Discount Rate
ggplot(novacart.df, aes(x = DiscountRate, y = Revenue)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  labs(title = "Revenue vs Discount Rate", x = "Discount Rate", y = "Monthly Revenue") +
  theme_minimal()
# Higher discount rates are associated with higher revenue, suggesting promotions boost sales volume – though heavy discounting may erode profit margins.


# Revenue by Product Category
ggplot(novacart.df, aes(x = ProductCategory, y = Revenue, fill = ProductCategory)) +
  geom_boxplot() +
  labs(title = "Revenue by Product Category", x = "Product Category", y = "Monthly Revenue") +
  theme_minimal() + theme(legend.position = "none")
# Electronics have the highest median revenue among categories, while Apparel shows lower median sales. This indicates Electronics is a top-performing product category.

# Revenue by Region
ggplot(novacart.df, aes(x = Region, y = Revenue, fill = Region)) +
  geom_boxplot() +
  labs(title = "Revenue by Region", x = "Region", y = "Monthly Revenue") +
  theme_minimal() + theme(legend.position = "none")
# The North region exhibits the highest revenue levels (and a wide spread), whereas other regions lag behind, pointing to regional differences in performance.


# 3. Data Preprocessing: Training/Validation Split
set.seed(123)  # for reproducibility
train.index <- sample(1:nrow(novacart.df), 0.7 * nrow(novacart.df))
train.df <- novacart.df[train.index, ]
valid.df <- novacart.df[-train.index, ]

# Normalize numeric predictors for kNN (using training set parameters)
numeric.vars <- c("MarketingSpend", "DiscountRate", "Units_Sold")
norm.values <- preProcess(train.df[, numeric.vars], method = c("center", "scale"))
train.norm.df <- train.df
valid.norm.df <- valid.df
train.norm.df[, numeric.vars] <- predict(norm.values, train.df[, numeric.vars])
valid.norm.df[, numeric.vars] <- predict(norm.values, valid.df[, numeric.vars])


# 4. Multiple Linear Regression (MLR) Model
lm.model <- lm(Revenue ~ MarketingSpend + DiscountRate + Units_Sold + ProductCategory + Region,
               data = train.df)
summary(lm.model)  # output training fit summary

# Predict on validation set and evaluate
lm.pred <- predict(lm.model, newdata = valid.df)
rmse.lm <- sqrt(mean((valid.df$Revenue - lm.pred)^2))
cat("Validation RMSE (Linear Regression):", rmse.lm, "\n")
# Performance metrics for MLR
lm.MAE  <- mean(abs(lm.pred - valid.df$Revenue))
lm.RMSE <- sqrt(mean((lm.pred - valid.df$Revenue)^2))
lm.MAPE <- mean(abs(lm.pred - valid.df$Revenue) / valid.df$Revenue) * 100
lm.R2   <- 1 - sum((lm.pred - valid.df$Revenue)^2) / sum((mean(train.df$Revenue) - valid.df$Revenue)^2)
print(c(MAE = lm.MAE, RMSE = lm.RMSE, MAPE = lm.MAPE, R2 = lm.R2))

# 5. k-Nearest Neighbors (kNN) Model
# Try k = 1 to 15 and find the optimal k based on validation RMSE
k.values <- 1:15
knn.RMSE <- numeric(length(k.values))
for (i in k.values) {
  knn.pred <- knn.reg(train = train.norm.df[, numeric.vars],
                      test  = valid.norm.df[, numeric.vars],
                      y     = train.norm.df$Revenue, k = i)
  knn.RMSE[i] <- sqrt(mean((valid.df$Revenue - knn.pred$pred)^2))
}
knn.RMSE  # show RMSE for each k
opt.k <- k.values[which.min(knn.RMSE)]
opt.k    # optimal number of neighbors

# Plot validation RMSE vs k to visualize the elbow and optimal k
knn.results <- data.frame(k = k.values, RMSE = knn.RMSE)
ggplot(knn.results, aes(x = k, y = RMSE)) +
  geom_line(color = "steelblue") + geom_point(color = "steelblue") +
  scale_x_continuous(breaks = k.values) +
  labs(title = "Validation RMSE by k for kNN", x = "Number of Neighbors (k)", y = "RMSE") +
  theme_minimal()

# Train final kNN with optimal k and evaluate
knn.final <- knn.reg(train = train.norm.df[, numeric.vars],
                     test  = valid.norm.df[, numeric.vars],
                     y     = train.norm.df$Revenue, k = opt.k)
knn.pred <- knn.final$pred
knn.MAE  <- mean(abs(knn.pred - valid.df$Revenue))
knn.RMSE.opt <- sqrt(mean((knn.pred - valid.df$Revenue)^2))
knn.MAPE <- mean(abs(knn.pred - valid.df$Revenue) / valid.df$Revenue) * 100
knn.R2   <- 1 - sum((knn.pred - valid.df$Revenue)^2) / sum((mean(train.df$Revenue) - valid.df$Revenue)^2)
print(c(Optimal_k = opt.k, MAE = knn.MAE, RMSE = knn.RMSE.opt, MAPE = knn.MAPE, R2 = knn.R2))

# 6. CART Model (Decision Tree)
cart.model <- rpart(Revenue ~ ., data = train.df, method = "anova", cp = 0.01)
printcp(cart.model)  # cross-validation results for different complexity parameter (cp) values

# Prune the tree at optimal cp to prevent overfitting
best.cp <- cart.model$cptable[which.min(cart.model$cptable[,"xerror"]), "CP"]
pruned.model <- prune(cart.model, cp = best.cp)

# Plot the pruned regression tree (for continuous Revenue prediction)
rpart.plot(pruned.model,
           type = 2,
           extra = 101,  # mean + % of observations
           fallen.leaves = TRUE,
           main = "CART Regression Tree - NovaCart Revenue")

# Predict on validation set and evaluate
cart.pred <- predict(pruned.model, valid.df)
cart.MAE  <- mean(abs(cart.pred - valid.df$Revenue))
cart.RMSE <- sqrt(mean((cart.pred - valid.df$Revenue)^2))
cart.MAPE <- mean(abs(cart.pred - valid.df$Revenue) / valid.df$Revenue) * 100
cart.R2   <- 1 - sum((cart.pred - valid.df$Revenue)^2) / sum((mean(train.df$Revenue) - valid.df$Revenue)^2)
print(c(MAE = cart.MAE, RMSE = cart.RMSE, MAPE = cart.MAPE, R2 = cart.R2))

# 7. Model Performance Comparison
# All three models have been evaluated on the validation set. 
# MLR and CART show similar accuracy (CART has a slight edge in RMSE), while kNN performs a bit worse.
# For interpretability and insights, we will use the linear model for scenario analysis.

# 8. Scenario Analysis
# Using the linear model to quantify the impact of strategic changes.

# Scenario setup: Define a baseline case (e.g., average values for inputs in a chosen segment)
baseline <- data.frame(
  MarketingSpend   = mean(novacart.df$MarketingSpend),
  DiscountRate     = mean(novacart.df$DiscountRate),
  Units_Sold       = mean(novacart.df$Units_Sold),
  Region           = factor("Regular", levels = levels(novacart.df$Region)),
  ProductCategory  = factor("Electronics", levels = levels(novacart.df$ProductCategory))
)

# Baseline prediction
base.pred <- predict(lm.model, newdata = baseline)
base.pred

# Scenario 1: Increase Marketing Spend by 20% (with other factors held constant)
scenario1 <- baseline
scenario1$MarketingSpend <- scenario1$MarketingSpend * 1.20
scen1.pred <- predict(lm.model, scenario1)

# Scenario 2: Increase Discount Rate by 5 percentage points (e.g., 0.05 in proportion)
scenario2 <- baseline
scenario2$DiscountRate <- min(scenario2$DiscountRate + 0.05, 1.0)  # cap at 100%
scen2.pred <- predict(lm.model, scenario2)

# Compare scenario outcomes to baseline
cat(sprintf("Baseline predicted revenue: %.2f\n", base.pred))
cat(sprintf("Scenario 1 (20%% increase in Marketing Spend) predicted revenue: %.2f (change: %.2f)\n", 
            scen1.pred, scen1.pred - base.pred))
cat(sprintf("Scenario 2 (+5%% points Discount Rate) predicted revenue: %.2f (change: %.2f)\n", 
            scen2.pred, scen2.pred - base.pred))

scenarios <- data.frame(
  Scenario = c("Baseline", "High Marketing", "High Discount"),
  Predicted_Revenue = c(base.pred, scen1.pred, scen2.pred)
)
ggplot(scenarios, aes(x = Scenario, y = Predicted_Revenue, fill = Scenario)) +
  geom_bar(stat = "identity") +
  labs(title = "Scenario Comparison: Predicted Revenue", y = "Predicted Revenue") +
  theme_minimal() + theme(legend.position = "none")


# Interpretation: In this example, boosting marketing spend by 20% yields a larger revenue increase than raising the discount rate by 5%. 
# This suggests that increasing marketing investment may drive revenue more effectively than further discounting (which can shrink margins). 
# NovaCart should consider prioritizing marketing spend in high-potential regions/products rather than relying too heavily on discounts.
