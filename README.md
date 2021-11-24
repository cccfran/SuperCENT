
<!-- README.md is generated from README.Rmd. Please edit that file -->

# SuperCENT

<!-- badges: start -->
<!-- badges: end -->

This package implements the Supervised Centrality Estimation (SuperCENT)
methodology in the paper: [Network regression and supervised centrality
estimation](https://jh-cai.com/docs/SuperCENT.pdf).

-   `two_stage()` produces the centrality estimation and regression
    coefficient estimation of the commonly used two-stage procedure.
-   `supercent()` produces SuperCENT estimate of the centralities and
    the regression coefficients for a fixed *λ*.
-   `cv.supercent()` produces SuperCENT estimate of the centralities and
    the regression coefficients with *K*-fold cross-validation.

## Installation

You can install the development version of SuperCENT from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("cccfran/SuperCENT")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(SuperCENT)

set.seed(1)
# generate network and (X, y)
n <- 2^8
d <- 1
beta0 <- c(1,3,5)
beta_u <- 2^6
beta_v <- 2^0
sigmaa <- 2^{0}
sigmay <- 2^{-6}

U <- matrix(rnorm(n), nrow = n)
U_norm <- sqrt(colSums(U^2))
U <- sweep(U, 2, U_norm, "/") * sqrt(n)

V <- matrix(rnorm(n), nrow = n)
V_norm <- sqrt(colSums(V^2))
V <- sweep(V, 2, V_norm, "/") * sqrt(n)

p <- length(beta0) 
X <- matrix(rnorm(n*(p-1)), nrow = n, ncol = p - 1, byrow = F)
X <- cbind(1, X)

y <- X %*% beta0 + beta_u * U + beta_v * V + rnorm(n, sd = sigmay)
A <- d * U %*% t(V) + matrix(rnorm(n^2, sd = sigmaa), nrow = n)

# SuperCENT oracle
## supercent(A, X, y, l = n*sigmay^2/sigmaa^2)

# SuperCENT CV
ret <- cv.supercent(A, X, y, lrange = 2^4, gap = 2, folds = 10)
## summary table
confint(ret)
## confidence interval
confint(ret, ci = T)
```
