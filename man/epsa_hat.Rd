% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utilities.R
\name{epsa_hat}
\alias{epsa_hat}
\title{Estimate \eqn{\sigma_a}}
\usage{
epsa_hat(ret, multi_rank = TRUE)
}
\arguments{
\item{ret}{A fitted network regression object}

\item{multi_rank}{Logical; include additional low-rank structure if available (default \code{TRUE})}
}
\value{
Estimated \eqn{\sigma_a}
}
\description{
Computes the standard deviation of residual noise in \code{A}
}
\seealso{
\code{\link{epsa_hat_oracle}}, \code{\link{epsy_hat}}
}
