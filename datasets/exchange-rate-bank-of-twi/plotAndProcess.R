chps <- c(31,53, 79, 135, 175, 225, 280)

data  <- read.table("./data/exchange-rate-twi-may-1970-aug-1.csv", sep = ",", header = TRUE)
data$exchangerate <- as.numeric(data$Exchange.Rate.TWI..May.1970...Aug.1995.)

plot(data$exchangerate, type = 'b')
abline(v=chps)

write.table(data$exchangerate, file="./data/sigTwi.dat", row.names = FALSE, col.names = FALSE)
write.table(chps, file="./data/changesTwi.dat", row.names = FALSE, col.names = FALSE)

png("./data/sigWithChangesTwi.PNG")
plot(data$exchangerate, type = 'b')
abline(v=chps)
dev.off()