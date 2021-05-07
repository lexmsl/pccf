set.seed(112)
x <- c(rnorm(100), rnorm(100) + 3.0, rnorm(100))
w <- 20
indFunc <- c(rep(0, 100-w),
             rep(1, 2*w),
             rep(0, 100-2*w),
             rep(1, 2*w),
             rep(0, 100-w)
             ) 
plot(x, type = 'l')
lines(indFunc*5, col="red", lw=2)
abline(v=c(100, 200), col="blue",lw=2)

write.table(x, file = "./artsig1.dat", col.names = FALSE, row.names = FALSE)
write.table(indFunc, file = "./indfunc1.dat", col.names = FALSE, row.names = FALSE)

png(filename = "./artsig1.PNG", width = 640, height=480, units = "px")
plot(x, type = 'l')
lines(indFunc*5, col="red", lw=2)
abline(v=c(100, 200), col="blue",lw=2)
dev.off()

