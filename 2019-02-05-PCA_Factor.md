---
layout: post
title:  "PCA "
date:   2019-02-05 23:59:00 +0100
categories: r PCA FactorAnalysis Analysis
---

### PCA(Principal Component Anlaysis)


#### PCA for hearing test result

The dataset is about hearing test. Variables are combination of frequency and ear side(L/R). Observable variables are 100 cases of dB information that participants can hear the variable frequencies. 


#####1. Explore Data

```{r read data, echo=TRUE}
data <- read.csv("/Users/keonhoseo/Documents/Q2/STAT 630/Assignment 1/audio.csv")
data <- data[,-1]
dim(data)

unscaled_data <- data

data=scale(data)
data=as.data.frame(data)
```

The audio csv file is consist of 100 rows and 8 columns. I consider this 8 columns as dimensions of data and the purpose of PCA analysis is to dimish the dimensions by linear combination for simplifying analysis. 


#####2. Define the problem in terms of PCs

```{r echo=TRUE}
sigma <- var(data)
sigma

vars=diag(sigma)
percentvars=vars/sum(vars)
```

PCA finds uncorrelated linear combinations of observed variables that explain maximal variance, because the maximal variance explains the character(feature) of data most. A covariance matrix of variables is useful to find the linear combinatoins since it represents covariances between two variables as a matrix in multivariate data.

Value of covariance(i,k) represents covariance value between i and kth variables and a diagonal of matrix is variances of variables. 


#####3. Compute all Eigenvalues/Eigenvectors 

```{r echo=TRUE}
eigenvalues=eigen(sigma)$values
eigenvectors=eigen(sigma)$vectors

eigenvalues
eigenvectors

y=as.matrix(data)%*%eigenvectors

sum(vars)
sum(eigenvalues)
```

Through Eigen decomposition of covariance matrix, I am able to identify directions and magnitudes of covariances. Eigenvectors and values are in pairs and the number of eigenvectors and values pairs exist is equals to number of dimensions that data has.

Covariance matrix's Eigenvectors represent directions of data scattered and Eigenvalues are telling how much variance there is in the data in that direction. So, the highest eigenvalue explaines the character of data most and paired eigenvector is the principal component.

Also eigenvectors represent relation between principal components and variables. PC1 has a negative relation with every variables, and PC2 has a negative relation with variables (3,4,7,8) while it owns a positive relation with the rest of variables.

From this I can assume PC1 is about general hearing and PC2 might be about a border of low/high frequency becuse variable number (3,4,7,8) are high frequencies. 

Thus total sum of variances in origin data and sum of eigenvalues are same, because PCA supposes not to lose explanation in variance, and accounts variance with a few variables. 


matrix y contains observable data's position value in each dimensions.


#####4. Check variance estimates of the pcs and all other properties
```{r echo=FALSE}
percentvars_pc = eigenvalues / sum(eigenvalues)
percentvars_pc
ts.plot(cbind(percentvars,percentvars_pc),col=c("blue","red"),xlab="ith vector",ylab="percent variance")
```

This result shows proportions of each eigenvalue in total sum of eigenvalues. Also it interprets the amount of variance that each principal component keeps. Thus, the first two principal components explain about 76.7% of total variance. 

The red line on the graph is a scree plot. It is easy to see the first two vector can explain high percent of variance.

#####5. Check correlation between components

```{r, echo=FALSE}
#eigenvectors[,1]
y1=as.matrix(data)%*%(eigenvectors[,1])
y2=as.matrix(data)%*%(eigenvectors[,2])
#cor(y1,data)
#eigenvectors[,1]*sqrt(eigenvalues[1])/sqrt(diag(vars))
```

```{r, echo=FALSE}
rho_pc <- cor(y)
rho_pc
```

This is a correlation matrix of principal components. Every value on the matrix is converging to 0, which means all of principal components are not correlated. 


#####6. Regression

```{r echo=FALSE}
set.seed(1002)
dv=rowSums(unscaled_data)+rnorm(100,mean=0,sd=10)
summary(lm(dv~as.matrix(y)))
summary(lm(dv~as.matrix(data)))
```

From regression result, I can see only differences in coefficients part, otherwise every regression results are same as origin data. This is because PCA presumes new linear line that accounts most variance through linear combination, and does not manipulate origin data.

L4000 and R2000 scores high p-value in regression analysis above. So, let's compare explanatory power between variables from origin data and Principal components. I pick variables L4000 and R2000 for origin data and the first two principal components(y1, y2) for PCA analysis.

```{r echo=FALSE}
## let's pick the best two
cor(dv,data)
summary(lm(dv~data$L4000+data$R2000))
summary(lm(dv~y1+y2))
```

Principal components regression analysis scored 0.9575 R-squared value while two variables from origin data marks 0.76 R-squared value. From this result, I can identify principal components has a higher explanatory power of data variances.

#####7. Draw a plot


```{r echo=FALSE}

plot(y1,y2)
text(y1,y2, labels = rownames(data), pos = 3)
```

Observable data are scattered by following principal components value.

Let's make a graph with third party package for better graph design.
```{r echo=FALSE}
library(factoextra)
pp=prcomp((data), scale = TRUE)  ## can change here
#summary(pp)
#fviz_eig(pp)

#fviz_pca_var(pp,col.var = "contrib",gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),repel = TRUE)    

fviz_pca_biplot(pp, repel = TRUE,col.var = "contrib",gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), col.ind = "#696969")

```


Observable data are on same position as the plain graph. 
This graph shows how much of variances are explained by principal components. X-axis, which is the first principal component, explains about 49.1% of variance and Y-axis, which is the second principal component, accounts about 20.2% of variance.

Also, the plot represents relation between variables and principal components. All variables have a negative relation with PC1 and Low frequency variables have a positive relation with PC2 while High frequency variables have a negative one with PC2. We already expect this on above eigenvectors value.


So, it seems PC1 is about general hearing and PC2 helps distinguishing hearing ability by Low and High frequency. We can interpret IDs(observable variables) on the right quadrants(1,4) has a good hearing ability in general. This is because if a person can hear sound on low dB, we assume the person has a good hearing. And then PC1 has negative relations with variables, so positive on X-axis means the person can hear every frequencies on low dB.

Observable variables on quadrant 4 might have a good hearing ability in general, but probably some difficulty in hearing high frequency compared to the variables on quadrant 1. Since Low/High frequency have a reverse relation with PC2, we can presume the observable variables in quadarnt 4 may listen high frequency on high dB.

On the contrary, IDs on the left quadrants(2,3) generally have a difficulty in hearing. Observable variables in quadrant 2 are more struggle to hear low frequency and the variables in quadrant 3 are in complication to listen high frequency.



#####8. Unstandardized

```{r echo=FALSE}
data <- read.csv("/Users/keonhoseo/Documents/Q2/STAT 630/Assignment 1/audio.csv")
data <- data[,-1]
sigma <- var(data)
eigenvalues=eigen(sigma)$values
eigenvectors=eigen(sigma)$vectors

y1=as.matrix(data)%*%(eigenvectors[,1])
y2=as.matrix(data)%*%(eigenvectors[,2])

set.seed(1002)
dv=rowSums(data)+rnorm(100,mean=0,sd=10)
summary(lm(dv~data$L4000+data$R2000))
summary(lm(dv~y1+y2))
```

Let's see a result with unstandardized data. R-squared value of PC1&2 is little bit different from standardized one, while origin data is same. Principal components are decided by variances. 


```{r echo=FALSE}
plot(y1,y2)
text(y1,y2, labels = rownames(data), pos = 3)
```

The plot is also distorted. Frequency 4000 on both sides' variance is much bigger than any other variables. So little difference on Frequency 4000 causes bigger differences on the results than other variables. Therefore, it is hard to interpret the plot correctly.





####Analysis on Grocery price

#####1. Explore Data

```{r echo=FALSE}
data <- read.csv("/Users/keonhoseo/Documents/Q2/STAT 630/Week 3/food.csv")
data

rownames(data)=data[,1]
data=data[,-1]

unscaled_data <- data

data=scale(data)
data=as.data.frame(data)

## Step 1 , Explore the data
dim(data)
var1=var(data$Bread)
var2=var(data$Hamburger)
var3=var(data$Butter)
var4=var(data$Apples)
var5=var(data$Tomato)

```

This dataset contains grocery price in U.S cities table. It has information about 24 U.S cities and 5 kinds of groceries. So I can consider it is 5 dimensions of data and it is difficult to identify comparable cities. The purpose of conducting PCA here is to explain the cities with a few variables, then we can compare the cities easily.


#####2. Define the problem in terms of PCs

```{r echo=FALSE}
sigma <- var(data)
sigma

vars=diag(sigma)
percentvars=vars/sum(vars)
```

The data is standardized above and compute covariances, because we want to know how 

#####3. Compute all Eigenvalues/Eigenvectors 

```{r echo=TRUE}
eigenvalues=eigen(sigma)$values
eigenvectors=eigen(sigma)$vectors

eigenvalues
eigenvectors

y=as.matrix(data)%*%eigenvectors
sum(vars)
sum(eigenvalues)
```

Decompose dataset with Eigenvectors and Eigenvalues. Eigenvectors mean directions of variables, so it indicates relations between variables and principal components. Eigenvalues reveal the amount of paired eigenvector can explain origin variances.

On here, all grocery has a negative relation wiht PC1, and only hamburger and tomato have positive relationship with PC2.

#####4. Check variance estimates of the pcs and all other properties

```{r echo=FALSE}
percentvars_pc = eigenvalues / sum(eigenvalues)
percentvars_pc
```

It reveals each eigenvalues' proportion out of total eigenvalue. First two principal components possess about 67.37% of the value, which means the two components clarify 67.37% of variances in data.

```{r echo=FALSE}
ts.plot(cbind(percentvars,percentvars_pc),col=c("blue","red"),xlab="ith vector",ylab="percent variance")
```

Graph a scree plot. Red line shows eigenvalues and the slope is getting slower after first 2 components. We can decide the first two components as variables for PCA.


#####5. Check correlation between components

```{r, echo=FALSE}
#eigenvectors[,1]
y1=as.matrix(data)%*%(eigenvectors[,1])
y2=as.matrix(data)%*%(eigenvectors[,2])
y3=as.matrix(data)%*%(eigenvectors[,3])

#cor(y1,data)
#eigenvectors[,1]*sqrt(eigenvalues[1])/sqrt(diag(vars))
```

```{r, echo=FALSE}
rho_pc <- cor(y)
rho_pc
```

This is correlations between principal components. Off-diagonal values are converging to 0, we don't have to worry about multicollinearity on the components.

#####6. Regression

Let's compare explanatory power for variances of origin variables and Principal components through regression analysis.

```{r, echo=FALSE}
set.seed(1002)
dv=rowSums(unscaled_data)+rnorm(24,mean=0,sd=10)
summary(lm(dv~as.matrix(data)))
summary(lm(dv~as.matrix(y)))
```

If we put complete dataset of origin and principal components into regression analysis, R-squared value and residual standard error are same in both analysis. This means component analysis explains whole varainces in origin data and PCA only finds linear lines through linear combination, not manipulating origin data.

```{r, echo=FALSE}
#cor(dv,data)
summary(lm(dv~y1+y2))
summary(lm(dv~data$Hamburger+data$Tomato))
```

Let's put Hamberguer and Tomato into regression as input and compare with principal components regression result. Hambugrger and Tomato combination get 0.7652 R-squared score, while the first two principal components scored 0.9259. So, we can say two of the PCA components explain more variances than any of two origin variables.


#####7. Draw a plot

```{r echo=FALSE}
plot(y1,y2)
text(y1,y2, labels = rownames(data), pos = 3)
```


Observable data are scattered by following principal components value. X-axis(PC1) explains about 48.8% of variances, and Y-axis(PC2) accounts for 18.6% of variances. 

Since every grocery variables have negative relations with PC1, positioning on left of X-axis means expensive in general. On the contrary, grocery price in the cities on right of X-axis should be cheaper. So livinging in Anchorage and Honolulu should cost a lot, and it makes sense.

On PC2(Y-axis), the most important variable on this is apple. Apple has a high negative relation with PC2. So we can expect the apple price in the cities on upper in Y-axis is cheaper than cities in bottom. Also, PC2 has positive relations with Hamburger and Tomato, which implies if Hamburger and Tomato are expensive the city should be on upper in Y-axis.

Through the PCA plot, it is easy to compare grocery prices in U.S cities and guesses living costs in the cities. 


#####8. Unstandardized

```{r echo=FALSE}
data <- read.csv("/Users/keonhoseo/Documents/Q2/STAT 630/Week 3/food.csv")
rownames(data)=data[,1]
data=data[,-1]
rho=cor(data)
## Step 2 - Define the problem in terms of principal components
sigma=var(data)
vars=diag(sigma)
percentvars=vars/sum(vars)

## Step 3 - Compute all the eigenvalues and eigenvectors in R

eigenvalues=eigen(sigma)$values
eigenvectors=eigen(sigma)$vectors

# define principal componenets
y1=as.matrix(data)%*%(eigenvectors[,1])
y2=as.matrix(data)%*%(eigenvectors[,2])

y=as.matrix(data)%*%eigenvectors

set.seed(1002)
dv=rowSums(data)+rnorm(24,mean=0,sd=10)
summary(lm(dv~y1+y2))
summary(lm(dv~data$Hamburger+data$Tomato))
```

The PCA regression results above are little bit different from the one that we saw in #6.Regression, while origin data regression is same. This is because I put unstandardized data as an input into regression. PCA works by finding maximum variances. So if data is not scaled, results should be distorted.

```{r echo=FALSE}
plot(y1,y2)
text(y1,y2, labels = rownames(data), pos = 4)
```

Also a plot is quite similar with standardized one, but little different from the scaled one.


###Factor Analysis


####Hearing dB/Frequency

#####1. Explore Data

```{r echo=FALSE}
data <- read.csv("/Users/keonhoseo/Documents/Q2/STAT 630/Assignment 1/audio.csv")
data <- data[,-1]
dim(data)

data=scale(data)
data=as.data.frame(data)

rho <- var(data)
rho
```

The audio data is also consists of 100 observable data and 8 columns. I will conduct factor analysis for the data this time. The purpose of conducting factor anaysis is to investigate variable relationships in data, so conductor can find concepts that are not easily discovered directly by crashing a large number of variables into a few interpretable underlying factors. Finding underlying factors is different from PCA. 


#####2. Compute the eigenvalues and eigenvactors of the correlation matrix
```{r echo=FALSE}
eigenvalues=eigen(rho)$values
eigenvalues
(eigenvalues)>1
m=2 

eigenvectors=eigen(rho)$vectors
```

On factor analysis, the eigenvalues is a measure of how much of the variance of the observed variables a factor explains. An eigenvalue >= 1 explains more variance than a single observed variable. For the dataset, eigenvalues that are larger than 1 are two, so I can expect two factors that explain the dataset.


#####3. Compute Estimated Factor Loadings
```{r echo=FALSE}
L=matrix(nrow=8,ncol=m)

for (j in 1:m){
L[,j]=sqrt(eigenvalues[j])*eigenvectors[,j]  
  }

L  # first column is Factor 1, 2nd column is factor 2
```

This is how compute estimated factor loadings. Factor loading is the relationship of each variable to the underlying factor. The higher the load the more relevant in defining the factor’s dimensionality. A negative loading indicates an inverse impact on the factor. So every variable has a negative relationship with factor 1. 

Also high frequency variables(#3,4,7,8) have a negative relation with the second factor, while other variables have a positive one. Freuqency 4000 variables on both L and R (#4,8) have a lower relation with factor 1 and a higher one with factor 2.

#####4. Compute Communality and unique variance
```{r echo=FALSE}
common=rowSums(L^2)
unique=1-common  ## this diagonal of error matrix

common
unique
```

A communality is the extent to which an variable correlates with all other variables, thus higher communality is better to identify a common character with other variables. Otherwise, it may struggle to load significantly on any factor.

In an opposite way, unique variance is the variance that is ‘unique’ to the variable and not shared with other variables. So, the greater uniqueness the lower relevance of the variable in the factor model.



#####5. Check the model to reproduce correlation
```{r echo=FALSE}
phi=diag(8)*unique

recreate=L%*%t(L)+phi
recreate

rho
```

I assume that the correlations between variables result from their sharing common underlying factors. Thus, it makes sense to try to estimate the correlations between variables from the correlations between factors and variables. The reproduced correlation matrix is obtained by multiplying the loading matrix by the transposed loading matrix. This results in calculating each reproduced correlation as the sum across factors of the products.

If difference between reproduced correlation matrix and correlation matrix is small, we can think the factor model is appropriate.

#####6. Create Residual Matrix and check appropriateness
```{r echo=FALSE}
residual=rho-recreate
residual  ## check to see if off-diagonal elements are "small"
```

This is a residual matrix that shows disparities.

```{r echo=FALSE}
sum(residual[lower.tri(residual)]^2)  ## sum of square of off-diagonal elements
```

This is sum of squared residual value, which should be small.

```{r echo=FALSE}
sum(eigenvalues[3:8]^2)  ## sum of square of non-used eigenvalues
```

And this is sum of squared non-used eigenvalues, which is the amount variance that is not explained by the factor model.


```{r echo=FALSE}

## Step 7  - Plot pairs of loadings to interpret factor loadings

plot(L[,1],L[,2],col=1:5,xlab="Loading 1",ylab="Loading 2")
text(L[,1],L[,2], pos = 3, names(data))

```

Draw a plot with factor loading 1, 2. It seems varialbes are divided as Low/High frequency. However, some of related factors are apart away. To identify factors clearly, we can rotate the plot.

#####7. Rotate 

```{r echo=FALSE}
library(psych)

## should reproduce our results
fit2 <- principal(data, nfactors=2, rotate="none")

fit2

fit <- principal(data, nfactors=2, rotate="varimax", covar=FALSE)
fit
```

Let's see why rotation is useful for identifying factors. From the principal components analysis above, we can see a difference in proportion variance in principal components while cumulative variance is same. That means PC2 has a higher explanatory power for variance when it is rotated. 

```{r echo=FALSE}
plot(fit$loadings[,1],fit$loadings[,2],col=1:5)
text(fit$loadings[,1],fit$loadings[,2],pos = 3, names(data))
```


On the plot without rotation, 500 and 2000 pairs of frequency are aside. However in a new plot with varimax rotataion, it is able to see more obvious factors.

Low/High frequency of both L and R sides are divided clearly. So somehow we can expect hearing frequency is important factor on the dataset, and Right ear always has less factor loading than Left ear on every frequencies.



#####8. Unstandardized


```{r echo=FALSE}
data <- read.csv("/Users/keonhoseo/Documents/Q2/STAT 630/Assignment 1/audio.csv")
data <- data[,-1]
rho <- cor(data)
eigenvalues=eigen(rho)$values
eigenvectors=eigen(rho)$vectors
m=2
L=matrix(nrow=8,ncol=m)

for (j in 1:m){
L[,j]=sqrt(eigenvalues[j])*eigenvectors[,j]  
}

common=rowSums(L^2)
unique=1-common  ## this diagonal of error matrix
phi=diag(8)*unique
recreate=L%*%t(L)+phi

plot(L[,1],L[,2],col=1:5,xlab="Loading 1",ylab="Loading 2")
text(L[,1],L[,2], pos = 3, names(data))

```

This is a plot of unstandardized data and it looks like same as the standardized one. This is because Factor Analysis uses correlation matrix. The thing that Anlaysis wants to know is relationship between variables, not a certain variable's variation. So, I can't find a difference between standardized and unstandardized.





####Survey on Pollution

Variables of this dataset are questions that were asked to people to know their opinion about pollution. Observable variables are the respondents' answers on the survey. Through factor analysis, I wish to classify questions as some groups that share coomon idea.

#####1. Explore & Cleaning Data
```{r echo=FALSE}
data <- read.csv("/Users/keonhoseo/Documents/Q2/STAT 630/Assignment 1/masst.csv")

data <- data[,20:39]
dim(data)
# Drop Missing Value
for(n in colnames(data))
{
  data <- data[which(data[[n]] != '.' & data[[n]] != '-'), ]
  data[[n]] <- as.numeric(as.character(data[[n]]))
}
dim(data)

data=scale(data)
data=as.data.frame(data)
```

Load and clean a dataset first. This dataset contains missing values. For this time, I decide to drop rows that include a missing value in any variable. Thus, it is 599 rows at first and it is shrinked as 550 rows after data cleaning. So, this is about 10% of loss. 

```{r echo=TRUE}
rho <- cor(data)
```

Calculate correlation between variables, because factor analysis wants to see similarities between variables we use correlation.

#####2. Compute the Eigenvalues and Eigenvectors of correlation
```{r echo=TRUE}
eigenvalues=eigen(rho)$values
eigenvalues
m=6

eigenvectors=eigen(rho)$vectors
#eigenvectors
```

Measure eigenvalues and eigenvectors.

For this dataset, I have 6 eigenvalues that are larger than 1. So I assume it has 6 latent factors.

#####3. Compute Estimated Factor Loadings

```{r echo=FALSE}
L=matrix(nrow=20,ncol=m)

for (j in 1:m){
  L[,j]=sqrt(eigenvalues[j])*eigenvectors[,j]  
}

L  # first column is Factor 1, 2nd column is factor 2
```

Compute factor loadings, it discloses relations between factor loadings and variables. So, if factor loading value is negative, it means the variable has a negative relation with the fact loading.

#####4. Compute Communality and unique variance

```{r echo=TRUE}
common=rowSums(L^2)
unique=1-common  ## this diagonal of error matrix

#common
#unique
```

#####5. Check the model to reproduce correlation
```{r echo=TRUE}
phi=diag(20)*unique

recreate=L%*%t(L)+phi
#recreate
```

#####6. Create Residual Matrix and check appropriateness
```{r echo=FALSE}
residual=rho-recreate
#residual  ## check to see if off-diagonal elements are "small"
```

```{r echo=FALSE}
sum(residual[lower.tri(residual)]^2)  ## sum of square of off-diagonal elements
```

Since the data matrix is large and it is difficult to see communality, unique and residual, let's see sum of squared of off-diagonal elements. If the model is appropriate, the result should be small and it is 0.966148.

```{r echo=FALSE}
sum(eigenvalues[7:20]^2)  ## sum of square of non-used eigenvalues
```

Also compute sum of square eigenvalues that are not used on this analysis. Eigenvalues explain the amount of the factor model explain variances, so sum of unused eigenvalues should be small too.

```{r echo=FALSE}

## Step 7  - Plot pairs of loadings to interpret factor loadings

plot(L[,1],L[,2],col=1:5,xlab="Loading 1",ylab="Loading 2")
text(L[,1],L[,2], pos = 3, names(data))

```

Draw a lot based on Loading 1 and 2. I can identify some factor groups, but most of groups are hard to recognize.

Let's rotate a plot by third party package and see difference when the plot is rotated by 'varimax'.

#####7. Rotate 

```{r echo=FALSE}
library(psych)

## should reproduce our results
fit2 <- principal(data, nfactors=6, rotate="none")

fit2

fit <- principal(data, nfactors=6, rotate="varimax")
fit
```

Important part on the analysis above is Proportion Var and Cumulative Var. Total variance that is explained by 6 principal components is same, 0.55. However proportion variance of each component is little different.

Proportion Variance

None    rotation: 0.15 0.10 0.10 0.08 0.06 0.05

Varimax rotation: 0.11 0.11 0.10 0.09 0.08 0.07

Compared to none rotation analysis, later PCs(4-6) cares higher variance.
For this reason, identifying factors becomes easier and let's check this on the new plot.

```{r echo=FALSE}
plot(fit$loadings[,1],fit$loadings[,2],col=1:5)
text(fit$loadings[,1],fit$loadings[,2],pos = 3, names(data))
```


It is easier to classify factors than the origin plot.

Through factor analysis, we can identify clusters on the plot above. Those might be divided as 5 groups.

Before mention about the clusters, it seems like factor loding 1 is accounting for personal ability/responsibility on pollution. The questions on X-axis's right are asking individual effort for preventing pollution and the quesitions on opposite side intend personal endeavor is useless.

On Y-axis, which is related to factor loading 2, factor groups are splited by thoughts about the seriousness of pollution. One group that is highly related ot factor loading 2 assumes resolving energy problem is more important than preserving pollution. On the contrary, most of other questions have a mark on conserving pollution.

Dividing question groups by axis are like above. Now then factor groups that stay on middle of graph should be handled. The groups in the middle of X-axis is not dealing with individual effort, it asks about endeavor of company, congress and president for pollution preserving. The factor in the top of middle in graph also mention about company, but it treats of presedence of resolving energy problem. Thus, this grop is identified easily.

Two groups, one includes (#21, 22, 25, 29, 38) and another one contains(#19, 20, 26, 32), are positioning quite similar area. Common part of the two groups is asking a role for preserving pollution. A disparity is the former one is dealing with company and technology, while the latter group is asking about a role of congress, president and the nation.

In conclusion, it is able to classify questions as 5 groups by projection of the first two principal components into reduced dimension through factor analysis.


#####8. Unstandardized

```{r echo=FALSE}
data <- read.csv("/Users/keonhoseo/Documents/Q2/STAT 630/Assignment 1/masst.csv")
data <- data[,20:39]
# Drop Missing Value
for(n in colnames(data))
{
  data <- data[which(data[[n]] != '.' & data[[n]] != '-'), ]
  data[[n]] <- as.numeric(as.character(data[[n]]))
}
eigenvalues=eigen(rho)$values
m=6
eigenvectors=eigen(rho)$vectors

L=matrix(nrow=20,ncol=m)

for (j in 1:m){
  L[,j]=sqrt(eigenvalues[j])*eigenvectors[,j]  
}

common=rowSums(L^2)
unique=1-common  ## this diagonal of error matrix

phi=diag(20)*unique
recreate=L%*%t(L)+phi
residual=rho-recreate

plot(L[,1],L[,2],col=1:5,xlab="Loading 1",ylab="Loading 2")
text(L[,1],L[,2], pos = 3, names(data))
```

Also this plot uses unstandardized data for Factor Analysis and it looks same as the standardized one. This is because Factor Analysis uses correlation between variables, not variances. 
