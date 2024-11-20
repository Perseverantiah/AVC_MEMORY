data <- read_csv("C:/Users/Ninette HOUKPONOU/Repertoire_python/Memoire/df2_encoder.csv")
shuffled_data= data[sample(1:nrow(data)), ]
data_train=shuffled_data[1:3981,]
data_test=shuffled_data[3982:4981,]
model=glm(stroke~.,family="poisson", data = data_train)
summary(model)

step(model,direction = "both",k=2)


data2=data_train[,16:21]
head(data2)
model2=glm(stroke~.,family="poisson", data = data2)
summary(model2)

step(model2,direction = "both",k=2)

cooks.distance(model2)[cooks.distance(model2) >1]
s2=sum(residuals(model2, type = "pearson")^2)
ddl=df.residual(model2)
p_value=1-pchisq(s2,ddl)
p_value=1-pchisq(deviance(model2),ddl)

predictions=predict.glm(model2,data_test,type="response")
comparaison_data=data.frame(predictions,obs=data_test$stroke)
head(comparaison_data)

library(MASS)
model3=glm.nb(stroke~., data = data2)
summary(model3)

step(model3,direction = "both",k=2)