---
layout: post
title:  "PCA"
date:   2019-02-05 23:59:00 +0100
categories: r PCA FactorAnalysis Analysis
---

### PCA(Principal Component Anlaysis)

####Analysis on Grocery price

#####1. Explore Data

{% highlight r %}
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
{% endhighlight %}


This dataset contains grocery price in U.S cities table. It has information about 24 U.S cities and 5 kinds of groceries. So I can consider it is 5 dimensions of data and it is difficult to identify comparable cities. The purpose of conducting PCA here is to explain the cities with a few variables, then we can compare the cities easily.


#####2. Define the problem in terms of PCs

{% highlight r %}
sigma <- var(data)
sigma

vars=diag(sigma)
percentvars=vars/sum(vars)
{% endhighlight %}

The data is standardized above and compute covariances, because we want to know how 

#####3. Compute all Eigenvalues/Eigenvectors 

{% highlight r %}
eigenvalues=eigen(sigma)$values
eigenvectors=eigen(sigma)$vectors

eigenvalues
eigenvectors

y=as.matrix(data)%*%eigenvectors
sum(vars)
sum(eigenvalues)
{% endhighlight %}


Decompose dataset with Eigenvectors and Eigenvalues. Eigenvectors mean directions of variables, so it indicates relations between variables and principal components. Eigenvalues reveal the amount of paired eigenvector can explain origin variances.

On here, all grocery has a negative relation wiht PC1, and only hamburger and tomato have positive relationship with PC2.

#####4. Check variance estimates of the pcs and all other properties

{% highlight r %}
percentvars_pc = eigenvalues / sum(eigenvalues)
percentvars_pc
{% endhighlight %}

It reveals each eigenvalues' proportion out of total eigenvalue. First two principal components possess about 67.37% of the value, which means the two components clarify 67.37% of variances in data.

{% highlight r %}
ts.plot(cbind(percentvars,percentvars_pc),col=c("blue","red"),xlab="ith vector",ylab="percent variance")
{% endhighlight %}

Graph a scree plot. Red line shows eigenvalues and the slope is getting slower after first 2 components. We can decide the first two components as variables for PCA.


#####5. Check correlation between components

{% highlight r %}
y1=as.matrix(data)%*%(eigenvectors[,1])
y2=as.matrix(data)%*%(eigenvectors[,2])
y3=as.matrix(data)%*%(eigenvectors[,3])
{% endhighlight %}



{% highlight r %}
rho_pc <- cor(y)
rho_pc
{% endhighlight %}


This is correlations between principal components. Off-diagonal values are converging to 0, we don't have to worry about multicollinearity on the components.

#####6. Regression

Let's compare explanatory power for variances of origin variables and Principal components through regression analysis.

{% highlight r %}
set.seed(1002)
dv=rowSums(unscaled_data)+rnorm(24,mean=0,sd=10)
summary(lm(dv~as.matrix(data)))
summary(lm(dv~as.matrix(y)))
{% endhighlight %}


If we put complete dataset of origin and principal components into regression analysis, R-squared value and residual standard error are same in both analysis. This means component analysis explains whole varainces in origin data and PCA only finds linear lines through linear combination, not manipulating origin data.

{% highlight r %}
#cor(dv,data)
summary(lm(dv~y1+y2))
summary(lm(dv~data$Hamburger+data$Tomato))
{% endhighlight %}

Let's put Hamberguer and Tomato into regression as input and compare with principal components regression result. Hambugrger and Tomato combination get 0.7652 R-squared score, while the first two principal components scored 0.9259. So, we can say two of the PCA components explain more variances than any of two origin variables.


#####7. Draw a plot

{% highlight r %}
plot(y1,y2)
text(y1,y2, labels = rownames(data), pos = 3)
{% endhighlight %}



Observable data are scattered by following principal components value. X-axis(PC1) explains about 48.8% of variances, and Y-axis(PC2) accounts for 18.6% of variances. 

Since every grocery variables have negative relations with PC1, positioning on left of X-axis means expensive in general. On the contrary, grocery price in the cities on right of X-axis should be cheaper. So livinging in Anchorage and Honolulu should cost a lot, and it makes sense.

On PC2(Y-axis), the most important variable on this is apple. Apple has a high negative relation with PC2. So we can expect the apple price in the cities on upper in Y-axis is cheaper than cities in bottom. Also, PC2 has positive relations with Hamburger and Tomato, which implies if Hamburger and Tomato are expensive the city should be on upper in Y-axis.

Through the PCA plot, it is easy to compare grocery prices in U.S cities and guesses living costs in the cities. 


#####8. Unstandardized

{% highlight r %}
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
{% endhighlight %}


The PCA regression results above are little bit different from the one that we saw in #6.Regression, while origin data regression is same. This is because I put unstandardized data as an input into regression. PCA works by finding maximum variances. So if data is not scaled, results should be distorted.

{% highlight r %}
plot(y1,y2)
text(y1,y2, labels = rownames(data), pos = 4)
{% endhighlight %}


Also a plot is quite similar with standardized one, but little different from the scaled one.


