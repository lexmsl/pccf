chps <- c(20, 40, 53, 69, 79, 95)

data  <- read.table("./data/number-of-earthquakes-per-year-m.csv", sep = ",", header = TRUE)

data$numOfEarthQuakes <- as.numeric(data$Number.of.earthquakes.per.year.magnitude.7.0.or.greater..1900.1998)

plot(data$numOfEarthQuakes, type = 'b')
abline(v=chps, col="red")

write.table(data$numOfEarthQuakes, file="./data/sigNumOfEarthQuakes.dat", row.names = FALSE, col.names = FALSE)

write.table(chps, file="./data/changesNumOfEarthQuakes.dat", row.names = FALSE, col.names = FALSE)
png("./data/sigWithChangesNumOfEarthQuakes.PNG")
plot(data$numOfEarthQuakes, type = 'b')
abline(v=chps, col="red")
dev.off()