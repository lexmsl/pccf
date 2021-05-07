chps <- c(90, 140, 255, 310, 420, 475, 585, 645, 750, 808, 910, 970, 1080, 1135)

data <- read.table("./data/internet-traffic-data-in-bits-fr.csv", sep = ",", header = TRUE)

data$traffic <- as.numeric(data$Internet.traffic.data..in.bits..from.a.private.ISP.with.centres.in.11.European.cities..The.data.corresponds.to.a.transatlantic.link.and.was.collected.from.06.57.hours.on.7.June.to.11.17.hours.on.31.July.2005..Hourly.data.)

plot(data$traffic, type = 'b')
abline(v=chps, col="red")

write.table(data$traffic, file="./data/sigInternetTraffic.dat", row.names = FALSE, col.names = FALSE)

write.table(chps, file="./data/changesInternetTraffic.dat", row.names = FALSE, col.names = FALSE)
png("./data/sigWithChangesInternetTraffic.PNG")
plot(data$traffic, type = 'b')
abline(v=chps, col="red")
dev.off()
