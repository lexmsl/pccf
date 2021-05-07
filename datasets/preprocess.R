#install.packages("jsonlite")
#install.packages("magrittr")
library(magrittr)
source("../src/Lib/R/fn_read_changes_and_signals.R")
r <- read_changes_and_signals("../src/config_datasets.json")
plt <- (function(x){plot(x, type = 'l')})
r$sig1 %>% plt

getSTLcomponents <- function(x){
  dec <- x %>%
    (function(x){ts(x, frequency = 2)}) %>%
    (function(x){stl(x, s.window = 2)})

  seasonal <- as.numeric(dec$time.series[,1])
  trend <- as.numeric(dec$time.series[,2])
  noise <- as.numeric(dec$time.series[,3])
  return(list(seasonal=seasonal, trend=trend, noise=noise))
}

d <- getSTLcomponents(r$sig1)

#dec <- r$sig1 %>%
#  (function(x){ts(x, frequency = 2)}) %>%
#  (function(x){stl(x, s.window = 2)})
#
#dec %>% plot
#dec$time.series[,1] %>% plot
#dec$time.series[,2] %>% plot
#dec$time.series[,3] %>% plot
#
#seasonal <- as.numeric(dec$time.series[,1])
#trend <- as.numeric(dec$time.series[,2])
#noise <- as.numeric(dec$time.series[,3])
#
#noise %>% plt

par(mfrow=c(2,1))
plot(r$sig1, type = 'l')
lines(d$trend, col="red")
plot(r$sig1 - d$noise, type = 'l')
