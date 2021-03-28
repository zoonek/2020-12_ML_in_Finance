LOG <- function (...) 
  cat(as.character(Sys.time()), " ", ..., "\n", sep = "")

## Code from http://www.mlfactor.com/

LOG( "Data" )

library(tidyverse)                      # Activate the data science package
library(lubridate)                      # Activate the date management package
load("raw/data_ml.RData")               # Load the data
data_ml <- data_ml %>% 
    filter(date > "1999-12-31",         # Keep the date with sufficient data points
           date < "2019-01-01") %>%
    arrange(stock_id, date)             # Order the data
data_ml[1:6, 1:6]                       # Sample values

features <- colnames(data_ml[3:95]) # Keep the feature's column names (hard-coded, beware!)
features_short <- c("Div_Yld", "Eps", "Mkt_Cap_12M_Usd", "Mom_11M_Usd", 
                    "Ocf", "Pb", "Vol1Y_Usd")

data_ml <- data_ml %>% 
    group_by(date) %>%                                   # Group by date
    mutate(R1M_Usd_C = R1M_Usd > median(R1M_Usd),        # Create the categorical labels
           R12M_Usd_C = R12M_Usd > median(R12M_Usd)) %>%
    ungroup() %>%
    mutate_if(is.logical, as.factor)

separation_date <- as.Date("2014-01-15")
training_sample <- filter(data_ml, date < separation_date)
testing_sample <- filter(data_ml, date >= separation_date)

stock_ids <- levels(as.factor(data_ml$stock_id)) # A list of all stock_ids
stock_days <- data_ml %>%                        # Compute the number of data points per stock
    group_by(stock_id) %>% summarize(nb = n()) 
stock_ids_short <- stock_ids[which(stock_days$nb == max(stock_days$nb))] # Stocks with full data
returns <- data_ml %>%                           # Compute returns, in matrix format, in 3 steps:
    filter(stock_id %in% stock_ids_short) %>%    # 1. Filtering the data
    dplyr::select(date, stock_id, R1M_Usd) %>%   # 2. Keep returns along with dates & firm names
    spread(key = stock_id, value = R1M_Usd)      # 3. Put in matrix shape
    
############################################################

LOG( "Models" )

library(xgboost)

sep_oos <- as.Date("2007-01-01")                            # Starting point for backtest
ticks <- data_ml$stock_id %>%                               # List of all asset ids
    as.factor() %>%
    levels()
N <- length(ticks)                                          # Max number of assets
t_oos <- returns$date[returns$date > sep_oos] %>%           # Out-of-sample dates 
    unique() %>%                                            # Remove duplicates
    as.Date(origin = "1970-01-01")                          # Transform in date format
Tt <- length(t_oos)                                         # Nb of dates, avoid T = TRUE
nb_port <- 2                                                # Nb of portfolios/stragegies
portf_weights <- array(0, dim = c(Tt, nb_port, N))          # Initialize portfolio weights
portf_returns <- matrix(0, nrow = Tt, ncol = nb_port)       # Initialize portfolio returns

weights_xgb <- function(train_data, test_data, features){ 
    train_features <- train_data %>% dplyr::select(features) %>% as.matrix()  # Indep. variable
    train_label <- train_data$R12M_Usd / exp(train_data$Vol1Y_Usd)            # Dep. variable
    ind <- which(train_label < quantile(train_label,0.2)|                     # Filter
                   train_label > quantile(train_label, 0.8))
    train_features <- train_features[ind, ]                                   # Filt'd features
    train_label <- train_label[ind]                                           # Filtered label
    train_matrix <- xgb.DMatrix(data = train_features, label = train_label)   # XGB format
    fit <- train_matrix %>% 
        xgb.train(data = .,                       # Data source (pipe input)
                  eta = 0.3,                      # Learning rate
                  objective = "reg:squarederror", # Number of random trees
                  max_depth = 4,                  # Maximum depth of trees
                  nrounds = 80,                   # Number of trees used
                  verbose = 0                     # No comments
        )
    xgb_test <- test_data %>%                     # Test sample => XGB format
        dplyr::select(features) %>% 
        as.matrix() %>%
        xgb.DMatrix()
    
    pred <- predict(fit, xgb_test)                # Single prediction
    w <- pred > median(pred)                      # Keep only the 50% best predictions
    w$weights <- w / sum(w)
    w$names <- unique(test_data$stock_id)
    return(w)                                     # Best predictions, equally-weighted
}

portf_compo <- function(train_data, test_data, features, j){ 
    if(j == 1){                                 # This is the benchmark
        N <- test_data$stock_id %>%             # Test data dictates allocation
            factor() %>% nlevels()
        w <- 1/N                                # EW portfolio
        w$weights <- rep(w,N)
        w$names <- unique(test_data$stock_id)   # Asset names
        return(w)
    }
    if(j == 2){                                 # This is the ML strategy.
        return(weights_xgb(train_data, test_data, features))
    }
}

LOG( "Loop: train the models on a moving window [LONG: 30 minutes]" )

m_offset <- 12                                          # Offset in months for buffer period
train_size <- 5                                         # Size of training set in years
for(t in 1:(length(t_oos)-1)){                          # Stop before last date: no fwd ret.!
    LOG( sprintf("[%s/%s]", t, length(t_oos)-1) )
    if(t%%12==0){print(t_oos[t])}                       # Just checking the date status
    train_data <- data_ml %>% filter(date < t_oos[t] - m_offset * 30,   # Roll window w. buffer
                                    date > t_oos[t] - m_offset * 30 - 365 * train_size)    
    test_data <- data_ml %>% filter(date == t_oos[t])   # Test sample  
    realized_returns <- test_data %>%                   # Computing returns via:
        dplyr::select(R1M_Usd)                          # 1M holding period!
    for(j in 1:nb_port){    
        temp_weights <- portf_compo(train_data, test_data, features, j) # Weights
        ind <- match(temp_weights$names, ticks) %>% na.omit()           # Index: test vs all
        portf_weights[t,j,ind] <- temp_weights$weights                  # Allocate weights 
        portf_returns[t,j] <- sum(temp_weights$weights * realized_returns) # Compute returns
    } 
}

LOG( "Performance" )

turnover <- function(weights, asset_returns, t_oos){ 
    turn <- 0
    for(t in 2:length(t_oos)){
        realised_returns <- returns %>% filter(date == t_oos[t]) %>% dplyr::select(-date)
        prior_weights <- weights[t-1,] * (1 + realised_returns) # Before rebalancing
        turn <- turn + apply(abs(weights[t,] - prior_weights/sum(prior_weights)),1,sum)
    }
    return(turn/(length(t_oos)-1))
}

perf_met <- function(portf_returns, weights, asset_returns, t_oos){ 
    avg_ret <- mean(portf_returns, na.rm = T)                     # Arithmetic mean 
    vol <- sd(portf_returns, na.rm = T)                           # Volatility
    Sharpe_ratio <- avg_ret / vol                                 # Sharpe ratio
    VaR_5 <- quantile(portf_returns, 0.05)                        # Value-at-risk
    turn <- 0                                                     # Initialisation of turnover
    for(t in 2:dim(weights)[1]){
        realized_returns <- asset_returns %>% filter(date == t_oos[t]) %>% dplyr::select(-date)
        prior_weights <- weights[t-1,] * (1 + realized_returns)
        turn <- turn + apply(abs(weights[t,] - prior_weights/sum(prior_weights)),1,sum)
    }
    turn <- turn/(length(t_oos)-1)                                # Average over time
    met <- data.frame(avg_ret, vol, Sharpe_ratio, VaR_5, turn)    # Aggregation of all of this
    rownames(met) <- "metrics"
    return(met)
}


perf_met_multi <- function(portf_returns, weights, asset_returns, t_oos, strat_name){
    J <- dim(weights)[2]              # Number of strategies 
    met <- c()                        # Initialization of metrics
    for(j in 1:J){                    # One very ugly loop
        temp_met <- perf_met(portf_returns[, j], weights[, j, ], asset_returns, t_oos)
        met <- rbind(met, temp_met)
    }
    row.names(met) <- strat_name      # Stores the name of the strat
    return(met)
}


asset_returns <- data_ml %>%                          # Compute return matrix: start from data
    dplyr::select(date, stock_id, R1M_Usd) %>%        # Keep 3 attributes 
    spread(key = stock_id, value = R1M_Usd)           # Shape in matrix format
asset_returns[is.na(asset_returns)] <- 0              # Zero returns for missing points

met <- perf_met_multi(portf_returns = portf_returns,  # Computes performance metrics
                      weights = portf_weights, 
                      asset_returns = asset_returns,
                      t_oos = t_oos,
                      strat_name = c("EW", "XGB_SR"))
met                                                   # Displays perf metrics


LOG( "Plot" )

library(lubridate) # Date management
library(cowplot)   # Plot grid management
g1 <- tibble(date = t_oos,  
      benchmark = cumprod(1+portf_returns[,1]),
      ml_based = cumprod(1+portf_returns[,2])) %>%
    gather(key = strat, value = value, -date) %>%
    ggplot(aes(x = date, y = value, color = strat)) + geom_line() +theme_grey()
g2 <- tibble(year = lubridate::year(t_oos),  
             benchmark = portf_returns[,1],
             ml_based = portf_returns[,2]) %>%
    gather(key = strat, value = value, -year) %>%
    group_by(year, strat) %>%
    summarise(avg_return = mean(value)) %>%
    ggplot(aes(x = year, y = avg_return, fill = strat)) + 
  geom_col(position = "dodge") + theme_grey()
plot_grid(g1,g2, nrow = 2)

############################################################

LOG( "Performance after 2016-01-01" )

DATE1 <- "2016-01-01"
rownames(portf_returns) <- as.character(t_oos)
r <- portf_returns[ t_oos > DATE1, 2 ]
r <- r[ r != 0 ]

w <- portf_weights[,2,]
rownames(w) <- as.character(t_oos)
w <- w[ t_oos > DATE1, ]

turnover <- rowSums( abs( apply( w, 2, diff ) ) )
turnover <- turnover[ - length(turnover) ]

LOG( "Save the results" )

res <- data.frame( t(c(
  CAGR                    = mean(r) * 12,
  `Annualized Volatility` = sd(r) * sqrt(12),
  `Information Ratio`     = mean(r) / sd(r) * sqrt(12),
  Turnover                = mean(turnover)
) ), check.names=FALSE )
write.csv(res, "results/mlfactor_perf.csv", row.names=FALSE)

LOG( "Done." )

