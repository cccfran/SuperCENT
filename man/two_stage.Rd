% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utilities.R
\name{two_stage}
\alias{two_stage}
\title{The two-stage procedure}
\usage{
two_stage(
  A,
  X,
  y,
  r = 1,
  scaled = 1,
  mode = "uv",
  weights = rep(1, length(y)),
  ...
)
}
\arguments{
\item{A}{The adjacency matrix of the input network}

\item{X}{The design matrix}

\item{y}{The response vector}

\item{r}{The rank of the input adjacency matrix}

\item{scaled}{Scale \eqn{u} and \eqn{v} of norm \eqn{\sqrt{n}}}

\item{weights}{The weight vector for each observation in (X,y)}
}
\value{
Output a \code{two_stage} object
\describe{
  \item{u}{The estimated hub centrality}
  \item{v}{The estimated authority centrality}
  \item{beta}{The scaled estimated regression coeffcients}
  \item{coefficients}{The original estimated regression coeffcients without scaling.}
  \item{residuals}{The residuals of the regression}
  \item{fitted.values}{The predicted response}
  \item{epsa}{The estimated \eqn{\sigma_a}}
  \item{epsy}{The estimated \eqn{\sigma_y}}
  \item{A}{The adjacency matrix of the input network}
  \item{X}{The input design matrix}
  \item{y}{The input response}
  \item{method}{The estimation method: two_stage}
  \item{...}{Auxiliary output from \code{lm.fit}}
}
}
\description{
The two-stage procedure first estimates the centralities
then regress the outcome of interest to the estimated 
centralities and other covariates.
}
\examples{
n <- 100
p <- 3
sigmaa <- 1
sigmay <- 1e-5
A <- matrix(rnorm(n^2, sd = sigmaa), nrow = n)
X <- matrix(rnorm(n*p), nrow = n, ncol = p)
y <- rnorm(n, sd = sigmay)
ret <- two_stage(A, X, y)
}
