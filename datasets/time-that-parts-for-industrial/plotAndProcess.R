chps <- c(19, 39, 58, 84)

data  <- read.table("./data/time-that-parts-for-industrial-p.csv", sep = ",", header = TRUE)

data$timesNeeded <- as.numeric(data$X..time.that.parts.for.industrial.project.available.when.needed..weekly.)

plot(data$timesNeeded, type = 'b')
abline(v=chps, col="red")

write.table(data$timesNeeded, file="./data/sigTimesNeeded.dat", row.names = FALSE, col.names = FALSE)

write.table(chps, file="./data/changesParts.dat", row.names = FALSE, col.names = FALSE)
png("./data/sigWithChangesPartsNeeded.PNG")
plot(data$timesNeeded, type = 'b')
abline(v=chps, col="red")
dev.off()