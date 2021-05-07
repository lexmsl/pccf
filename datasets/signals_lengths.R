library(magrittr)
source("../src/Lib/R/fn_read_changes_and_signals.R")
r <- read_changes_and_signals("../src/config_datasets.json")
plt <- (function(x){plot(x, type = 'l')})
r$sig1 %>% plt

lenTwi              <- length(r$sig1)
lentInternetTraffFr <- length(r$sig2)
lenLakeErie         <- length(r$sig3)
lenEarthqakes       <- length(r$sig4)
lenParts            <- length(r$sig5)
lenSunspotNumbers   <- length(r$sig6)


dat <- data.frame(names=c("twi", "internetTraffFr", "lakeErie", "earthqakes", "parts", "sunspotNumbers"),
                  lengths=c(lenTwi, lentInternetTraffFr, lenLakeErie, lenEarthqakes, lenParts, lenSunspotNumbers))
dat
write.table(dat,file="./sigLengths.csv", row.names = FALSE,sep=",")
