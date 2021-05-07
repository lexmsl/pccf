chps_detailed <- c(8, 17,26, 32, 37, 44, 49, 61, 67, 70, 76, 81, 88)
chps <- c(19,49, 57, 81)

data  <- read.table("./data/wolfer-sunspot-numbers-1770-to-1.csv", sep = ",", header = TRUE)
data$sunspotnumber <- as.numeric(data$Wolfer.sunspot.numbers..1770.to.1869)

plot(data$sunspotnumber, type = 'b')
abline(v=chps)

write.table(data$sunspotnumber, file="./data/sigSunspotNumber.dat", row.names = FALSE, col.names = FALSE)

write.table(chps, file="./data/changes.dat", row.names = FALSE, col.names = FALSE)

png("./data/sigWithChanges.PNG")
plot(data$sunspotnumber, type = 'b')
abline(v=chps)
dev.off()