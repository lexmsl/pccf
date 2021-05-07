chps <- c(65,105,120,150, 190, 250, 350, 455, 525)

data  <- read.table("./data/monthly-lake-erie-levels-1921-19.csv", sep = ",", header = TRUE)

data$monthlylevel <- as.numeric(data$Monthly.Lake.Erie.Levels.1921...1970.)

plot(data$monthlylevel, type = 'b')
abline(v=chps, col="red")

write.table(data$monthlylevel, file="./data/sigLakeLevel.dat", row.names = FALSE, col.names = FALSE)

write.table(chps, file="./data/changesErieLake.dat", row.names = FALSE, col.names = FALSE)
png("./data/sigWithChangesErieLake.PNG")
plot(data$monthlylevel, type = 'b')
abline(v=chps, col="red")

dev.off()