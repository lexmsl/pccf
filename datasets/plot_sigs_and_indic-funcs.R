library(magrittr)
library(jsonlite)
s = fromJSON("../src/config_datasets.json")
source("../src/Lib/R/fn_read_changes_and_signals.R")
r <- read_changes_and_signals("../src/config_datasets.json")
plt <- (function(x){plot(x, type = 'l')})
r$sig1 %>% plt

s$ind2
ind1 <- read.table("./src-scala-construct-ideal-pccfs-output/step13-scala-ind-func-twi.dat")$V1
ind2 <- read.table("./src-scala-construct-ideal-pccfs-output/step13-scala-ind-func-internet.dat")$V1
ind3 <- read.table("./src-scala-construct-ideal-pccfs-output/step13-scala-ind-func-lake.dat")$V1
ind4 <- read.table("./src-scala-construct-ideal-pccfs-output/step13-scala-ind-func-earthquakes.dat")$V1
ind5 <- read.table("./src-scala-construct-ideal-pccfs-output/step13-scala-ind-func-parts.dat")$V1
ind6 <- read.table("./src-scala-construct-ideal-pccfs-output/step13-scala-ind-func-sunspot.dat")$V1

scale01<-function(x){(x - min(x))/ (max(x)-min(x))}

plot(scale01(r$sig1), type = 'l')
abline(v=r$chps1)
lines(ind1, col = "red")

plot(scale01(r$sig2), type = 'l')
abline(v=r$chps2)
lines(ind2, col = "red")

plot(scale01(r$sig3), type = 'l')
abline(v=r$chps3)
lines(ind3, col = "red")

plot(scale01(r$sig4), type = 'l')
abline(v=r$chps4)
lines(ind4, col = "red")

plot(scale01(r$sig5), type = 'l')
abline(v=r$chps5)
lines(ind5, col = "red")

plot(scale01(r$sig6), type = 'l')
abline(v=r$chps6)
lines(ind6, col = "red")
