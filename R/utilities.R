#' Compute the Euclidean norm of a vector
#'
#' @param x A numeric vector
#' @return The Euclidean norm \eqn{\|x\| = \sqrt{\sum x_i^2}}
#' @examples 
#' vec_norm(c(3, 4))  # returns 5
vec_norm <- function(x) {sqrt(sum(x^2))}


#' Spectral norm of the difference between outer products
#' 
#' Computes the spectral norm of the difference between the outer products of two vectors \eqn{x x^\top - y y^\top}.
#' Optionally normalizes the vectors before computing the difference.
#' 
#' @param x A numeric vector
#' @param y A numeric vector
#' @param scale Logical; whether to normalize \code{x} and \code{y} to unit norm before computing outer products
#' @return The spectral norm (largest singular value) of the difference between the two outer products
#' @examples 
#' x <- c(1, 2)
#' y <- c(2, 4)
#' spec_norm_diff(x, y)
spec_norm_diff <- function(x, y, scale = T) {
  if(length(x) != length(y)) stop('x and y are of different length')
  if(length(x) != 1 & scale) {
    x <- x/sqrt(sum(x^2))
    y <- y/sqrt(sum(y^2))
  }
  norm(x %*% t(x) - y %*% t(y), "2")
  # mat <- x %*% t(x) - y %*% t(y)
  # if(sum(mat) == 0) {
  #   ret <- 0
  # } else {
  #   ret <- irlba::irlba(mat, nu = 1)
  # }
  # ret
}

#' Initialize a result object for network regression models
#'
#' Constructs an empty list with initialized slots for use in centrality-based regression procedures.
#' 
#' @param A The adjacency matrix
#' @param X The design matrix
#' @param y The response vector
#' @return A list with initialized slots for centrality estimation, regression coefficients, and meta info
#' @examples 
#' ret <- ret_constructor(matrix(0, 2, 2), matrix(1, 2, 1), c(1, 2))
ret_constructor <- function(A, X, y) {
  ret <- NULL
  
  ret$A <- A
  ret$X <- X
  ret$y <- y
  
  ret$u <- NA
  ret$v <- NA
  ret$d <- NA
  
  ret$beta <- NA
  
  ret$epsy <- NA
  ret$epsa <- NA
  
  ret$method <- NA
  ret$iter <- NA
  ret$l <- NA
  
  ret
}

#' Experimental: Shift unobserved nodes in adjacency matrix (for semi-supervised estimation)
#'
#' As part of a semi-supervised estimation strategy, this function reorders the adjacency matrix
#' such that observed nodes appear first and unobserved nodes (those with missing outcomes) 
#' are shifted to the bottom-right block. This is useful for block-decomposable estimators or 
#' conditional eigendecomposition techniques.
#' 
#' @param A A square adjacency matrix representing the network
#' @param weights A logical vector of length equal to \code{nrow(A)} where \code{TRUE} indicates observed nodes and \code{FALSE} unobserved nodes
#' @return A reordered adjacency matrix with unobserved nodes placed last
#' @examples 
#' A <- matrix(1:16, 4, 4)
#' weights <- c(TRUE, TRUE, FALSE, FALSE)
#' A_shift_unobserved_last(A, weights)
#' 
#' @seealso \code{\link{unobserved_shift_back}}
#' @note This function is experimental and part of a semi-supervised network estimation feature.
A_shift_unobserved_last <- function(A, weights) {
  
  n <- nrow(A)
  
  A_ori <- A
  unobs <- which(!weights)
  obs <- which(weights == 1)
  n_obs <- sum(weights == 1)
  
  A[1:n_obs, 1:n_obs] <- A_ori[obs, obs]
  A[(1+n_obs):n, (1+n_obs):n] <- A_ori[unobs, unobs]
  A[1:n_obs, (1+n_obs):n] <- A_ori[obs, unobs]
  A[(1+n_obs):n, 1:n_obs] <- A_ori[unobs, obs]
  
  A
}


#' Experimental: Revert vector indexing after semi-supervised adjacency shift
#'
#' This function reverts the node indexing of any vector that was aligned with a shifted adjacency matrix
#' (as produced by \code{A_shift_unobserved_last}). It is used to restore original order of node-level 
#' quantities (e.g., eigenvectors, centrality scores) after estimation on the reordered matrix.
#'
#' @param xx A numeric vector aligned to the shifted adjacency matrix
#' @param weights A logical vector used for the original shift (same as in \code{A_shift_unobserved_last})
#' @return A numeric vector reordered back to match the original node indexing
#' @examples 
#' xx <- c(10, 20, 30, 40)
#' weights <- c(TRUE, TRUE, FALSE, FALSE)
#' unobserved_shift_back(xx, weights)
#' 
#' @seealso \code{\link{A_shift_unobserved_last}}
#' @note This function is experimental and intended for semi-supervised learning pipelines where
#' only a subset of nodes have observed responses.
unobserved_shift_back <- function(xx, weights) {
  
  n <- length(xx)
  
  unobs <- which(!weights)
  obs <- which(weights == 1)
  n_obs <- sum(weights == 1)
  
  xx_tmp <- xx
  xx_tmp[obs] <- xx[1:n_obs]
  xx_tmp[unobs] <- xx[(1+n_obs):n]
  
  xx_tmp
}

#' The two-stage procedure
#' 
#' The two-stage procedure first estimates the centralities
#' then regress the outcome of interest to the estimated 
#' centralities and other covariates.
#' 
#' @param A The adjacency matrix of the input network
#' @param X The design matrix 
#' @param y The response vector
#' @param r The rank of the input adjacency matrix
#' @param scaled Scale \eqn{u} and \eqn{v} of norm \eqn{\sqrt{n}}
#' @param weights The weight vector for each observation in (X,y)
#' @return Output a \code{two_stage} object
#' \describe{
#'   \item{u}{The estimated hub centrality}
#'   \item{v}{The estimated authority centrality}
#'   \item{beta}{The scaled estimated regression coeffcients}
#'   \item{coefficients}{The original estimated regression coeffcients without scaling.}
#'   \item{residuals}{The residuals of the regression}
#'   \item{fitted.values}{The predicted response}
#'   \item{epsa}{The estimated \eqn{\sigma_a}}
#'   \item{epsy}{The estimated \eqn{\sigma_y}}
#'   \item{A}{The adjacency matrix of the input network}
#'   \item{X}{The input design matrix}
#'   \item{y}{The input response}
#'   \item{method}{The estimation method: two_stage}
#'   \item{...}{Auxiliary output from \code{lm.fit}}
#' }
#' @examples 
#' n <- 100
#' p <- 3
#' sigmaa <- 1
#' sigmay <- 1e-5
#' A <- matrix(rnorm(n^2, sd = sigmaa), nrow = n)
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' y <- rnorm(n, sd = sigmay)
#' ret <- two_stage(A, X, y)
two_stage <- function(A, X, y, r = 1, scaled = 1, mode = "uv", weights = rep(1, length(y)), ...) {
  
  n = nrow(A)
  k = ifelse(is.null(X), 0, ncol(X))
  n_obs = sum(weights == 1)
  
  # move the unobserved to the end
  A_ori <- A
  if(n_obs < n) A <- A_shift_unobserved_last(A_ori, weights)
  
  multi_rank <- FALSE
  if(r > 1) multi_rank = TRUE
  
  # SVD
  svda <- irlba::irlba(A, nu = r, nv = r)
  uu <- svda$u[,1]
  vv <- svda$v[,1]
  d <- svda$d[1]
  
  # OLS
  if(mode == "uv") {
    ret_two_stage <- lm.fit(cbind(X, uu[1:n_obs], vv[1:n_obs]), y)
  } else if (mode == "u") {
    ret_two_stage <- lm.fit(cbind(X, uu[1:n_obs], 0), y)
    ret_two_stage$coefficients[ncol(X)+2] <- 0
  } else if (mode == "v") {
    ret_two_stage <- lm.fit(cbind(X, 0, vv[1:n_obs]), y)
    ret_two_stage$coefficients[ncol(X)+1] <- 0
  }
  ret_two_stage$beta <- ret_two_stage$coefficients
  
  if(scaled) {
    uu <- uu*sqrt(n); vv <- vv*sqrt(n); d <- d/n
    ret_two_stage$beta[k+1] <- ret_two_stage$beta[k+1]/sqrt(n)
    ret_two_stage$beta[k+2] <- ret_two_stage$beta[k+2]/sqrt(n)
  }
  
  if(n_obs < n) {
    uu <- unobserved_shift_back(uu, weights)
    vv <- unobserved_shift_back(vv, weights)
  }
  
  ret_two_stage$u <- uu
  ret_two_stage$v <- vv
  ret_two_stage$d <- d
  
  ret_two_stage <- adjust_sign(ret_two_stage)
  
  ret_two_stage$u_distance <- NA
  ret_two_stage$method <- "two_stage"
  ret_two_stage$mode <- mode
  
  ret_two_stage$A <- A_ori
  ret_two_stage$X <- X
  ret_two_stage$y <- y
  
  ret_two_stage$epsa <- epsa_hat(ret_two_stage, multi_rank = multi_rank)
  ret_two_stage$epsy <- epsy_hat(ret_two_stage)
  
  ret_two_stage$iter <- NA
  ret_two_stage$l <- NA
  
  if(multi_rank) {
    U_perp <- svda$u[, -1, drop=F]
    V_perp <- svda$v[, -1, drop=F]
    d_perp <- svda$d[-1]
    
    ret_two_stage$multi_rank <- list(d_perp = d_perp/n,
                                     U_perp = U_perp*sqrt(n),
                                     V_perp = V_perp*sqrt(n),
                                     A_perp = U_perp %*% diag(d_perp) %*% t(V_perp))
    
  }
  
  class(ret_two_stage) <- "two_stage"
  
  ret_two_stage
}


#' Estimate the optimal scaling parameter for latent confounding
#'
#' Computes the theoretically motivated scaling factor \eqn{\ell} that balances
#' the contribution of latent confounding effects and observed noise. This is based 
#' on a two-stage estimation of centralities and regression coefficients. The procedure 
#' adjusts for the proportion of observed outcomes.
#'
#' @param A The adjacency matrix of the input network
#' @param X The design matrix
#' @param y The response vector
#' @param weights A logical vector indicating which nodes have observed outcomes
#' @param r The rank for low-rank approximation (default is 1)
#' @param multi_rank Logical; whether to estimate \eqn{\sigma_a} using multiple singular values (default is \code{TRUE})
#' 
#' @return A numeric scalar \eqn{\ell}, representing the optimal scale for debiasing or penalization in downstream estimation
#' 
#' @examples
#'
lopt_estimate <- function(A, X, y, weights, r = 1, multi_rank = TRUE) {
  
  n <- length(y)
  n_obs <- sum(weights == 1)
  
  ret_two_stage <- two_stage(A, X, y, r=r, weights = weights)
  sigmayhat2 <- epsy_hat(ret_two_stage)^2
  sigmaahat2 <- epsa_hat(ret_two_stage, multi_rank = multi_rank)^2
  
  l <- n*sigmayhat2/sigmaahat2*n/n_obs
  
  l
}

#' SuperCENT with fixed \eqn{\lambda}
#' 
#' The SuperCENT methodology that simultaneously solves 
#' the centrality estimation and regression given a fixed \eqn{\lambda}.
#' 
#' @param A The input network
#' @param X The design matrix 
#' @param y The response vector
#' @param l The tuning parameter of the penalty
#' @param folds The number of fold for cross-validation
#' @param tol The precision tolerance to stop
#' @param max_iter The maximum iteration
#' @param weights The weight vector for each observation in (X,y)
#' @param verbose Output detailed message at different levels
#' @return Output a \code{supercent} object
#' \describe{
#'   \item{d}{The estimated \eqn{d}}
#'   \item{u}{The estimated hub centrality}
#'   \item{v}{The estimated authority centrality}
#'   \item{beta}{The scaled estimated regression coeffcients}
#'   \item{l}{The tuning parameter \eqn{\lambda}}
#'   \item{residuals}{The residuals of the regression}
#'   \item{fitted.values}{The predicted response}
#'   \item{epsa}{The estimated \eqn{\sigma_a}}
#'   \item{epsy}{The estimated \eqn{\sigma_y}}
#'   \item{A}{The adjacency matrix of the input network}
#'   \item{X}{The input design matrix}
#'   \item{y}{The input response}
#'   \item{iter}{The grid of the tuning parameter}
#'   \item{max_iter}{The maximum iteration}
#'   \item{u_distance}{The sequence of differences of \eqn{\hat{u}} between 
#'   the two consecutive iterations}
#'   \item{method}{The estimation method: supercent}
#' }
#' @examples 
#' n <- 100
#' p <- 3
#' sigmaa <- 1
#' sigmay <- 1e-5
#' A <- matrix(rnorm(n^2, sd = sigmaa), nrow = n)
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' y <- rnorm(n, sd = sigmay)
#' ret <- supercent(A, X, y)
supercent <- function(A, X, y, l = NULL, tol = 1e-4, max_iter = 200, mode = "uv",
                      weights = rep(1, length(y)), verbose = 0,
                      r = 1, multi_rank = TRUE, rand_int = FALSE, ...) {
  
  n = nrow(A)
  k = ifelse(is.null(X), 0, ncol(X))
  n_obs = sum(weights == 1)
  
  # move the unobserved to the end
  A_ori <- A
  if(n_obs < n) {
    print("Semi-SuperCENT")
    A <- A_shift_unobserved_last(A_ori, weights)
  }
  
  if(r == 1) multi_rank <- FALSE
  
  if(is.null(l)) l <- lopt_estimate(A, X, y, weights, r = r, multi_rank = multi_rank)
  
  if(rand_int) {
    svda <- svd(A, nu=1, nv=1)
    u <- svda$u[,1] + rnorm(n,0,10/sqrt(n))
    v <- svda$v[,1] + rnorm(n,0,10/sqrt(n))
    u <- u/norm(u, "2")*sqrt(n)
    v <- v/norm(v, "2")*sqrt(n)
    d <- svda$d[1] / n
    ret <- lr_rand_int(A, X, y, d, u, v, l, tol, max_iter, verbose)
  } else {
    if(mode == "uv") {only_u <- F; only_v <- F}
    if(mode == "u") {only_u <- T; only_v <- F}
    if(mode == "v") {only_u <- F; only_v <- T}
    ret <- lr(A, X, y, l, tol, max_iter, verbose, only_u = only_u, only_v = only_v)
  }
  
  # Adjust sign
  ret <- adjust_sign(ret)
  
  # shift the order back
  if(n_obs < n) {
    ret$u <- unobserved_shift_back(ret$u, weights)
    ret$v <- unobserved_shift_back(ret$v, weights)
  }
  
  # multi-rank
  if(multi_rank) {
    svda <- svd(A - ret$d * ret$u %*% t(ret$v))
    U_perp <- svda$u[,1:(r-1),drop=F]
    V_perp <- svda$v[,1:(r-1),drop=F]
    d_perp <- svda$d[1:(r-1)]
    
    ret$multi_rank <- list(d_perp = d_perp/n,
                           U_perp = U_perp*sqrt(n),
                           V_perp = V_perp*sqrt(n),
                           A_perp = U_perp %*% diag(d_perp, nrow = r-1) %*% t(V_perp))
    
  }
  
  ret$A <- A_ori
  ret$X <- X
  ret$y <- y
  ret$mode <- mode
  
  ret$epsa <- epsa_hat(ret, multi_rank = multi_rank)
  ret$epsy <- epsy_hat(ret)
  
  class(ret) <- "supercent"
  
  ret
}

#' Experimental: Randomized initialization for SuperCENT optimization
#'
#' This function runs the \code{supercent} procedure multiple times with random initializations
#' and returns the result that achieves the smallest final centrality estimation error.
#' This can help mitigate poor local optima in non-convex estimation problems.
#'
#' @param A The adjacency matrix of the input network
#' @param X The design matrix
#' @param y The response vector
#' @param l The regularization parameter (optional; if \code{NULL}, defaults are used internally)
#' @param tol Convergence tolerance for stopping criterion (default \code{1e-4})
#' @param max_iter Maximum number of iterations per run (default \code{200})
#' @param weights A logical or numeric vector indicating which observations are used (default is all observed)
#' @param verbose Verbosity level; \code{0} means silent, \code{1} prints convergence info (default \code{0})
#' @param rand_times Number of random restarts to try (default \code{30})
#' @param ... Additional arguments passed to \code{supercent}
#'
#' @return A \code{supercent} object corresponding to the best result (lowest final \code{u_distance})
#'
#' @examples
rand.supercent <- function(A, X, y, l = NULL, tol = 1e-4, max_iter = 200, 
                            weights = rep(1, length(y)), verbose = 0, rand_times = 30, ...) {
  
  ret <- NULL
  ret$obj <- 1e12
  
  for(i in 1:rand_times) {
    ret_tmp <- supercent(A, X, y, l, tol, max_iter, weights, verbose, TRUE)
    if(tail(ret_tmp$u_distance, 1) < ret$obj) ret <- ret_tmp
    ret$obj <- tail(ret_tmp$u_distance, 1)
  }
  
  ret
}



#' SuperCENT k-fold cross-validation
#' 
#' The SuperCENT methodology that simultaneously solves
#' the centrality estimation and regression using
#' k-fold cross-validation to choose the tuning parameter \eqn{\lambda}.
#' 
#' @param A The input network
#' @param X The design matrix 
#' @param y The response vector
#' @param l The initial tuning parameter
#' @param lrange The search range of the tuning parameter
#' @param gap The search gap of the tuning parameter
#' @param folds The number of fold for cross-validation
#' @param tol The precision tolerance to stop
#' @param max_iter The maximum iteration
#' @param weights The weight vector for each observation in (X,y)
#' @param verbose Output detailed message at different levels
#' @return Output a \code{cv.supercent} object
#' \describe{
#'   \item{d}{The estimated \eqn{d}}
#'   \item{u}{The estimated hub centrality}
#'   \item{v}{The estimated authority centrality}
#'   \item{beta}{The scaled estimated regression coeffcients}
#'   \item{l}{The tuning parameter \eqn{\lambda}}
#'   \item{residuals}{The residuals of the regression}
#'   \item{fitted.values}{The predicted response}
#'   \item{epsa}{The estimated \eqn{\sigma_a}}
#'   \item{epsy}{The estimated \eqn{\sigma_y}}
#'   \item{A}{The adjacency matrix of the input network}
#'   \item{X}{The input design matrix}
#'   \item{y}{The input response}
#'   \item{l_sequence}{The grid of the tuning parameter}
#'   \item{beta_cvs}{The estimated regression coefficients of \code{l_sequence}}
#'   \item{mse_cv}{The cross-validation MSEs of \code{l_sequence}}
#'   \item{cv_index}{The fold indices (X,y)}
#'   \item{iter}{The grid of the tuning parameter}
#'   \item{max_iter}{The maximum iteration}
#'   \item{u_distance}{The sequence of differences of \eqn{\hat{u}} between 
#'   the two consecutive iterations}
#'   \item{method}{The estimation method: supercent}
#' }
#' @examples 
#' n <- 100
#' p <- 3
#' sigmaa <- 1
#' sigmay <- 1e-5
#' A <- matrix(rnorm(n^2, sd = sigmaa), nrow = n)
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#' y <- rnorm(n, sd = sigmay)
#' ret <- cv.supercent(A, X, y)
cv.supercent <- function(A, X, y, 
                         l = NULL, lrange = 2^4, gap = 2, 
                         folds = 10, tol = 1e-4, max_iter = 200, 
                         mode = "uv",
                         weights = rep(1, length(y)), verbose = 0, 
                         r = 1, multi_rank = TRUE, ...) {
  
  n = nrow(A)
  k = ifelse(is.null(X), 0, ncol(X))
  n_obs = sum(weights == 1)
  
  # move the unobserved to the end
  A_ori <- A
  if(n_obs < n) A <- A_shift_unobserved_last(A_ori, weights)
  
  if(r == 1) multi_rank <- FALSE
  
  if(is.null(l)) l <- lopt_estimate(A, X, y, weights, r = r, multi_rank = multi_rank)
  
  lmin <- l/lrange*(2^gap)
  # lmin <- l
  lmax <- l*lrange
  
  if(mode == "uv") {only_u <- F; only_v <- F}
  if(mode == "u") {only_u <- T; only_v <- F}
  if(mode == "v") {only_u <- F; only_v <- T}
  
  if(n_obs < n) {
    print("Semi-SuperCENT")
    ret <- cv_lr_2(A, X, y, lmin, lmax, gap, tol, max_iter, folds, verbose) 
  } else {
    print("SuperCENT")
    ret <- cv_lr(A, X, y, lmin, lmax, gap, tol, max_iter, folds, verbose, early_stopping = F, only_u = only_u, only_v = only_v) 
  }
  
  # Adjust sign
  ret <- adjust_sign(ret)
  
  # shift the order back
  if(n_obs < n) {
    ret$u <- unobserved_shift_back(ret$u, weights)
    ret$v <- unobserved_shift_back(ret$v, weights)
  }
  
  # multi-rank
  if(multi_rank) {
    svda <- svd(A - ret$d * ret$u %*% t(ret$v))
    U_perp <- svda$u[,1:(r-1),drop=F]
    V_perp <- svda$v[,1:(r-1),drop=F]
    d_perp <- svda$d[1:(r-1)]
    
    ret$multi_rank <- list(d_perp = d_perp/n,
                           U_perp = U_perp*sqrt(n),
                           V_perp = V_perp*sqrt(n),
                           A_perp = U_perp %*% diag(d_perp, nrow = r-1) %*% t(V_perp))
    
  }
  
  ret$A <- A_ori
  ret$X <- X
  ret$y <- y
  ret$mode <- mode
  
  ret$epsa <- epsa_hat(ret, multi_rank = multi_rank)
  ret$epsy <- epsy_hat(ret)
  
  class(ret) <- "cv.supercent"
  
  ret
}

#' Adjust signs of centrality vectors and coefficients
#'
#' Resolves the sign ambiguity of estimated centrality vectors \code{u} and \code{v}
#' by making the largest-magnitude entry positive, and adjusting associated 
#' coefficients in \code{beta} accordingly.
#'
#' @details 
#' The vectors \code{u}, \code{v} are only identifiable up to sign, but their products
#' with covariates or coefficients (e.g., \eqn{u \cdot \hat{\beta}_u}) are uniquely identifiable.
#' This adjustment ensures sign consistency across runs or for interpretation when needed.
#'
#' @param ret A network regression output
#' @return The network regression output with consistently adjusted signs
#'
#' @examples 
#' ret <- list(u = c(-2, -1), beta = c(1, -3))
#' adjust_sign(ret)
adjust_sign <- function(ret) {
  
  if(is.null(ret$v)) {
    k <- length(ret$beta) - 1
    
    u_change_condition <- (sign(ret$u[which.max(abs(ret$u))]) < 0)
    
    if(u_change_condition) {
      ret$u <- -ret$u
      ret$beta[k+1] <- -ret$beta[k+1]
    }
  } else {
    k <- length(ret$beta) - 2
    
    u_condition <- (max(abs(ret$u)) > max(abs(ret$v)))
    u_change_condition <- (sign(ret$u[which.max(abs(ret$u))]) < 0)
    
    v_condition <- !u_condition
    v_change_condition <- (sign(ret$v[which.max(abs(ret$v))]) < 0)
    
    if( (u_condition & u_change_condition) | 
        (v_condition & v_change_condition)) {
      
      ret$u <- -ret$u; ret$v <- -ret$v;
      ret$beta[k+1] <- -ret$beta[k+1]
      ret$beta[k+2] <- -ret$beta[k+2]
      
    }
  }
  
  ret
}

#' Construct oracle output for network regression
#'
#' This function outputs an oracle estimator,
#' where true centralities and regression coefficients are known.
#' Useful for simulation studies and validation against estimated procedures.
#'
#' @param A The original adjacency matrix
#' @param X The design matrix
#' @param y The response vector
#' @param d A vector of singular values (first entry used as primary rank)
#' @param U Matrix of left singular vectors (e.g., true \code{u} basis)
#' @param V Matrix of right singular vectors (e.g., true \code{v} basis)
#' @param beta0vec The true regression coefficient vector
#' @param beta_hat An estimated or debiased coefficient vector
#' @param epsa The true or estimated latent noise level for \code{A}
#' @param epsy The true or estimated noise level for \code{y}
#' @param A_hat A possibly denoised version of \code{A}
#' @param l The optimal or chosen regularization parameter
#' @param weights Logical vector indicating observed outcomes
#' @param method A character string indicating the estimation method name
#' @param multi_rank Logical; whether to include orthogonal components beyond rank-1
#' @param U_perp Orthogonal left singular vectors (if \code{multi_rank = TRUE})
#' @param V_perp Orthogonal right singular vectors (if \code{multi_rank = TRUE})
#' @param d_perp Singular values for orthogonal components (if \code{multi_rank = TRUE})
#'
#' @return A list with all inputs stored, and derived quantities such as residuals and low-rank components included
#'
#' @examples 
#' n <- 100; p <- 3
#' A <- matrix(rnorm(n^2), n)
#' X <- matrix(rnorm(n * p), n, p)
#' y <- rnorm(n)
#' U <- matrix(rnorm(n), n); V <- matrix(rnorm(n), n)
#' beta0vec <- runif(p + 2)
#' oracle(A, X, y, d = 1, U, V, beta0vec, beta_hat = beta0vec,
#'        epsa = 1, epsy = 0.1, A_hat = A, weights = rep(TRUE, n),
#'        method = "oracle", multi_rank = FALSE)
#'
#' @seealso \code{\link{ret_constructor}}, \code{\link{supercent}}, \code{\link{adjust_sign}}
#' @note This is intended primarily for simulation benchmarking and evaluation.

oracle <- function(A, X, y, d, U, V, beta0vec, beta_hat, epsa, epsy,
                   A_hat, l = NA,
                   weights = observed, method, multi_rank = TRUE,
                   U_perp, V_perp, d_perp) {
  
  unobs <- which(!weights)
  obs <- which(weights == 1)
  
  ret <- ret_constructor(A, X, y)
  
  ret$d <- d[1]
  ret$u <- U
  ret$v <- V
  ret$beta <- beta0vec
  ret$beta_hat <- beta_hat
  
  ret$residuals <- (cbind(X, U[obs,1], V[obs,1]) %*% beta0vec - y)
  
  df <- opt$n_train - length(beta0vec)
  
  ret$A <- A
  ret$A_hat <- A_hat
  ret$X <- X
  ret$y <- y
  
  ret$epsa <- epsa
  ret$epsy <- epsy
  
  ret$l <- l
  
  ret$method <- method
  
  # multi-rank
  if(multi_rank) {
    ret$multi_rank <- list(d_perp = d_perp,
                           U_perp = U_perp,
                           V_perp = V_perp,
                           A_perp = U_perp %*% diag(d_perp) %*% t(V_perp))
    
  }
  
  ret
}

#' Predict outcomes using two-stage network regression
#'
#' Applies the estimated coefficients from a two-stage network regression model to test data.
#'
#' @param A Full adjacency matrix containing both training and test nodes
#' @param X_test Covariate matrix for the test set
#' @param ret_two_stage A fitted object from \code{two_stage()}
#' @param scaled Logical; whether to scale test singular vectors to match training norm (default \code{TRUE})
#'
#' @return A list with predicted outcomes \code{y}, test centralities \code{u} and \code{v}
#'
#' @examples
#' # Assume last 20 rows/cols of A correspond to test nodes
#' pred <- predict_two_stage(A, X_test, ret)
#'
#' @seealso \code{\link{two_stage}}, \code{\link{predict_supervised}}
predict_two_stage <- function(A, X_test, ret_two_stage, scaled = 1) {
  N <- nrow(A)
  n_test <- nrow(X_test)
  n_train <- N - n_test
  pred_two_stage <- NULL
  svda <- svd(A, nu=1, nv=1)
  # svda <- irlba(A, nu=1, nv=1)
  pred_two_stage$u <- svda$u[(n_train+1):N,1,drop=F]
  pred_two_stage$v <- svda$v[(n_train+1):N,1,drop=F]
  if(scaled) {
    utrain_norm <- norm(svda$u[1:n_train,1,drop=F], "F") 
    vtrain_norm <- norm(svda$v[1:n_train,1,drop=F], "F")
    pred_two_stage$u <- svda$u[(n_train+1):N,1,drop=F] / utrain_norm * sqrt(n_train)
    pred_two_stage$v <- svda$v[(n_train+1):N,1,drop=F] / vtrain_norm * sqrt(n_train)
  }
  pred_two_stage$y <- cbind(X_test, pred_two_stage$u, pred_two_stage$v) %*% ret_two_stage$beta
  pred_two_stage
}


#' Predict using known centrality in two-stage model (oracle)
#'
#' Computes predictions using provided \code{u_test}, \code{v_test}.
#' Intended for oracle evaluation or simulation studies where ground truth is available.
#'
#' @param A Adjacency matrix (not used in this version, included for interface compatibility)
#' @param X_test Covariate matrix for the test set
#' @param ret_two_stage A fitted object from \code{two_stage()}
#' @param u_test Known test centrality vector (left)
#' @param v_test Known test centrality vector (right)
#' @param scaled Logical; ignored in current version (included for compatibility)
#'
#' @return A list with predicted outcomes \code{y}, and given \code{u}, \code{v}
#'
#' @note This is a redundant oracle version assuming known latent factors
#' @seealso \code{\link{predict_two_stage}}, \code{\link{oracle}}
predict_two_stage_oracle <- function(A, X_test, ret_two_stage, u_test, v_test, scaled = 1) {
  pred_two_stage <- NULL
  pred_two_stage$y <- cbind(X_test, u_test, v_test) %*% ret_two_stage$beta
  pred_two_stage$u <- u_test
  pred_two_stage$v <- v_test
  pred_two_stage
}

predict_supervised <- function(ret, A, X, weights) {
  
  n = nrow(A)
  n_obs = sum(weights == 1)
  
  # if(n_obs < n) A <- A_shift_unobserved_last(A, weights)
  
  svda <- irlba::irlba(A, nu = 1, nv = 1)
  uu <- svda$u[,1]
  vv <- svda$v[,1]
  d <- svda$d[1]
  
  if(sign(uu[1:n_obs] %*% ret$u[1:n_obs]) < 0) {
    uu <- -uu; vv <- -vv;
  }
  
  u_train_norm <- sqrt(sum(uu[1:n_obs]^2))
  v_train_norm <- sqrt(sum(vv[1:n_obs]^2))
  
  u_test <- uu[(n_obs+1):n]/u_train_norm*sqrt(n_obs); 
  v_test <- vv[(n_obs+1):n]/v_train_norm*sqrt(n_obs); 
  
  list(y_test = cbind(X, u_test, v_test) %*% ret$beta, 
       u2 = u_test, v2 = v_test)
  
}

K.mat <- function(m,n)
{
  x <- matrix(0,m*n,m*n)
  m0 <- 1:(m*n)
  n0 <- as.vector(t(matrix(m0,m,n)))
  
  arr.ind <- cbind(m0, n0)
  dims <- c(m*n,m*n)
  arr.indMat <- matrix(arr.ind, ncol = length(dims))
  idx1 <- cumprod(dims[-length(dims)])
  idx2 <- arr.indMat[, -1, drop = FALSE]-1
  idxRaw <- rowSums(idx1*idx2) + arr.indMat[, 1]
  
  x[idxRaw] <- 1
  
  return(x)
}

K.mat.sparse <- function(m, n) 
{
  m0 <- 1:(m*n)
  n0 <- as.vector(t(matrix(m0,m,n)))
  
  arr.ind <- cbind(m0, n0)
  dims <- c(m*n,m*n)
  arr.indMat <- matrix(arr.ind, ncol = length(dims))
  idx1 <- cumprod(dims[-length(dims)])
  idx2 <- arr.indMat[, -1, drop = FALSE]-1
  idxRaw <- rowSums(idx1*idx2) + arr.indMat[, 1]
  
  mn <- m*n
  x <- sparseMatrix(i = (idxRaw-1) %% mn + 1, j = (idxRaw-1) %/% mn + 1, x = 1)
  
  return(x)
}

#' Align signs of estimated coefficients with ground truth
#'
#' @param beta Estimated coefficient vector
#' @param beta0 Ground truth or reference coefficient vector
#' @return The input \code{beta} with signs aligned to \code{beta0} for the last two entries
#'
#' @examples
#' beta_sign(c(1, -2, -3), c(1, -2, 3))  # flips last entry
beta_sign <- function(beta, beta0)
{
  p <- length(beta0)
  beta[p-1] <- ifelse(beta[p-1]*beta0[p-1] < 0, -beta[p-1], beta[p-1])
  beta[p] <- ifelse(beta[p]*beta0[p] < 0, -beta[p], beta[p])
  
  beta
}

#' Align sign of estimated centrality with ground truth
#'
#' @param u Estimated latent vector
#' @param u0 Ground truth or reference latent vector
#' @return The input \code{u} with aligned sign
#'
#' @examples
#' u_sign(c(-1, -2), c(1, 2))  # flips sign
u_sign <- function(u, u0)
{
  sgn <- sign(c(t(u0)%*%u))
  u <- sgn*u
  
  u
}

#' Align signs of full network regression output with ground truth
#'
#'
#' @param ret A result object from \code{supercent}, \code{two_stage}, or similar
#' @param beta0vec Reference coefficient vector
#' @param u0 Reference left latent vector
#' @param v0 Reference right latent vector
#' @return The input \code{ret} with aligned \code{u}, \code{v}, \code{beta}, and optionally \code{beta_hat}
#'
#' @seealso \code{\link{beta_sign}}, \code{\link{u_sign}}, \code{\link{adjust_sign}}
ret_sign <- function(ret, beta0vec, u0, v0)
{
  if(!is.null(ret$beta_hat)) ret$beta_hat <- beta_sign(ret$beta_hat, beta0vec)
  ret$beta <- beta <- beta_sign(ret$beta, beta0vec)
  ret$u <- u_sign(ret$u, u0)
  ret$v <- u_sign(ret$v, v0)
  
  ret
}

#' Variance estimation for network regression parameters
#' 
#' @param X_train Covariate matrix for observed nodes
#' @param beta_u Estimated coefficient on centrality \code{u}
#' @param beta_v Estimated coefficient on centrality \code{v}
#' @param d Estimated leading singular value
#' @param u Estimated centrality vector (left)
#' @param v Estimated centrality vector (right)
#' @param epsa2 Estimated noise variance in the network (\code{A})
#' @param epsy2 Estimated noise variance in the outcome (\code{y})
#' @param l Regularization parameter (optional, used in low-rank models)
#' @param method Estimation method, one of \code{"lr"} ) (SuperCENT) or \code{"two_stage"}
#'
#' @return A numeric vector of estimated variances for \code{u}, \code{v}, \code{beta_u}, \code{beta_v}, and \code{beta_x}
var_mat <- function(X_train, beta_u, beta_v, d, u, v, epsa2, epsy2, l = NULL, method = "lr")
{
  n <- nrow(X_train)
  p <- ncol(X_train)
  W <- cbind(X_train, u, v)
  P <- W %*% solve(t(W)%*%W) %*% t(W)
  I <- diag(1, nrow = n)
  K <- K.mat.sparse(n, n)
  
  if(method == "two_stage") {
    c11 <- c21 <- matrix(0, nrow = n, ncol = n)
    c12 <- t(v)%x%(I-u%*%t(u)/n)/(d*n)
    c22 <- t(u)%x%(I-v%*%t(v)/n)/(d*n)
  } else if(method == "lr") {
    denom <- l*d^2 + beta_u^2 + beta_v^2
    # c11 <- beta_u / denom * (I - P)
    # c21 <- beta_v / denom * (I - P)
    # 
    # c12 <- t(v)%x%(I-u%*%t(u)) - (beta_u^2*t(v)%x%(I-P) + beta_u*beta_v*(t(u)%x%(I-P)) %*% K)/denom
    # c12 <- c12/d
    # c22 <- t(u)%x%(I-v%*%t(v))%*%K - (beta_u*beta_v*t(v)%x%(I-P) + beta_v^2*(t(u)%x%(I-P)) %*% K)/denom
    # c22 <- c22/d
    
    betauvmat <- matrix(c(beta_u^2,beta_u*beta_v,beta_u*beta_v, beta_v^2),nrow=2)
    c1 <- (diag(1, nrow=2*n) - betauvmat%x%(I-P)/denom)%*% rbind(beta_u*(I-P), beta_v*(I-P))/l/d^2
    c11 <- c1[1:n,] 
    c21 <- c1[(n+1):(2*n),]
    
    Mu <- (I-u%*%t(u)/n)
    Mv <- (I-v%*%t(v)/n)
    
    c2 <- (diag(1, nrow=2*n) - betauvmat%x%(I-P)/denom) %*% (l*d*rbind(t(v)%x%Mu/n, t(u)%x%(Mv/n)%*%K))/l/d^2
    c12 <- c2[1:n,]
    c22 <- c2[(n+1):(2*n),]
  }
  
  Px <- X_train %*% solve(t(X_train)%*%X_train) %*% t(X_train)
  tu <- (I-Px) %*% u
  tv <- (I-Px) %*% v
  c <- as.numeric(t(tu)%*%tu%*%t(tv)%*%tv - (t(tu)%*%tv)^2)
  
  # sanity check
  mul1 <- -t(tv)%*%tv%*%t(tu) + t(tu)%*%tv%*%t(tv)
  mul2 <- -t(tu)%*%tu%*%t(tv) + t(tu)%*%tv%*%t(tu)
  
  # matuv <- matrix(c(t(tu) %*% tu, t(tu) %*% tv, t(tu) %*% tv, t(tv) %*% tv), nrow = 2)
  # c341 = solve(matuv)%*%rbind(t(tu), t(tv))
  # 
  # c342 = -1/d*solve(matuv)%*%rbind(beta_v*(t(u)%x%(t(tu)%*%(I - v%*%t(v))))%*%K,
  #                                  beta_u*(t(v)%x%(t(tv)%*%(I - u%*%t(u)))))
  
  c31 = cbind(beta_u*mul1, beta_v*mul1, -mul1) %*% rbind(c11, c21, I) / c 
  c41 = cbind(beta_u*mul2, beta_v*mul2, -mul2) %*% rbind(c11, c21, I) / c 
  c32 = cbind(beta_u*mul1, beta_v*mul1) %*% rbind(c12, c22) / c
  c42 = cbind(beta_u*mul2, beta_v*mul2) %*% rbind(c12, c22) / c
  
  XXX <- solve(t(X_train)%*%X_train)%*%t(X_train)
  c5_l = cbind(-beta_u*XXX, -beta_v*XXX, -XXX%*%u, -XXX%*%v)
  c51 <- cbind(c5_l, XXX) %*% rbind(c11, c21, c31, c41, I)
  c52 <- c5_l %*% rbind(c12, c22, c32, c42)
  
  
  c_mat_ <- cbind(rbind(c11, c21, c31, c41, c51)*epsy2,
                  rbind(c12, c22, c32, c42, c52)*epsa2)
  c_mat <- cbind(rbind(c11, c21, c31, c41, c51),
                 rbind(c12, c22, c32, c42, c52))
  vars <- rowSums(c_mat_ * c_mat)
  
  vars[c(1:2*n, (2*n+3):length(vars), (2*n+1), (2*n+2))]
}

#' Confidence intervals for all network regression parameters
#'
#' Computes standard errors, test statistics, and p-values for estimated centrality components 
#' (\code{u}, \code{v}) and regression coefficients.
#'
#' @param ret A fitted network regression object (e.g., from \code{supercent})
#' @param alpha Confidence level (default \code{0.05})
#' @param method Estimation method used, one of \code{"lr"} or \code{"two_stage"}
#'
#' @return A data frame with parameter estimates, standard errors, t-statistics, and p-values
#' @seealso \code{\link{var_mat}}, \code{\link{epsa_hat}}, \code{\link{epsy_hat}}
confint_all <- function(ret, alpha = 0.05, method = "lr") {
  n <- length(ret$y)
  p <- ncol(ret$X)
  
  sigmay2 <- epsy_hat(ret)^2
  sigmaa2 <- epsa_hat(ret)^2
  
  l <- NULL
  if(!is.null(ret$l)) l <- ret$l
  
  vars <- var_mat(X_train = ret$X, beta_u = ret$beta[p+1], beta_v = ret$beta[p+2], 
                  d = ret$d, u = ret$u, v = ret$v, 
                  epsa2 = sigmaa2, epsy2 = sigmay2, l = l, method = method)
  sds <- sqrt(vars)
  
  est <- c(ret$u, ret$v, ret$beta)
  param <- c(paste0("u", rownames(ret$A)), paste0("v", colnames(ret$A)),
             "betau", "betav", paste0("betax", 1:p))
  # ci <- data.frame(param = c(paste0("u", 1:n), paste0("v", 1:n),
  #                            "betau", "betav", paste0("betax", 1:p)),
  #                  estimate = est,
  #                  ci_lwr = est - qnorm(1-alpha/2) * sds,
  #                  ci_upr = est + qnorm(1-alpha/2) * sds)
  
  tval <- est / sds
  pval <- 2*pnorm(-abs(tval))
  
  summmary_tbl <- data.frame(estimate = est, 
                             sds = sds, 
                             t = tval, 
                             p = pval)
  rownames(summmary_tbl) <- param
  
  summmary_tbl
}

#' Legacy: Oracle estimate of network noise level
#'
#' Computes \eqn{\epsilon_a} using ground truth \code{A0} and the fitted rank-1 structure from \code{u}, \code{v}, \code{d}.
#'
#' @param A0 True adjacency matrix
#' @param d Leading singular value
#' @param u Estimated centrality vector (left)
#' @param v Estimated centrality vector (right)
#'
#' @return Oracle estimate of \eqn{\epsilon_a}
epsa_hat_oracle <- function(A0, d, u, v) {
  n <- length(u)
  sqrt(sum((d * u%*%t(v) - A0)^2)/(n^2-1))
}


#' Estimate \eqn{\sigma_a}
#'
#' Computes the standard deviation of residual noise in \code{A}
#'
#' @param ret A fitted network regression object
#' @param multi_rank Logical; include additional low-rank structure if available (default \code{TRUE})
#'
#' @return Estimated \eqn{\sigma_a}
#'
#' @seealso \code{\link{epsa_hat_oracle}}, \code{\link{epsy_hat}}
epsa_hat <- function(ret, multi_rank = TRUE) {
  n <- length(ret$u)
  if(!is.null(ret$v)) {
    A_hat <- ret$d*ret$u%*%t(ret$v)
  } else {
    A_hat <- ret$d*ret$u%*%t(ret$u)
  }
  
  if(multi_rank) {
    A_hat <- A_hat + ret$multi_rank$A_perp
  }
  
  epsa2 <- sum((A_hat - ret$A)^2)/n^2
  
  sqrt(epsa2)
}

#' Estimate \eqn{\sigma_y}
#'
#' Computes the residual standard error \eqn{\epsilon_y}.
#'
#' @param ret A fitted network regression object
#'
#' @return Estimated \eqn{\epsilon_y}
#'
#' @seealso \code{\link{epsa_hat}}, \code{\link{confint_all}}
epsy_hat <- function(ret) {
  n <- length(ret$residuals)
  p <- length(ret$beta)
  
  sqrt(sum(ret$residuals^2)/(n - p))
}

#' Confidence interval and summary table of \eqn{\beta}
#' 
#' This function returns the confidence interval or 
#' the summary table of \code{two_stage}, \code{supercent}
#' and \code{cv.supercent} object.
#' 
#' @param ret A  \code{two_stage}, \code{supercent}
#' and \code{cv.supercent} object
#' @param alpha The level of type-I error
#' @param ci If TRUE, return the confidence interval;
#' if FALSE, return the summary table.
#' @return Output a data.frame of confidence interval
#' or summary table.
confint <- function(ret, alpha = 0.05, ci = F, multi_rank = F) {
  
  n <- length(ret$y)
  k <- ncol(ret$X)
  betahat <- ret$beta
  # for oracle
  if(!is.null(ret$beta_hat)) betahat <- ret$beta_hat
  
  A_perp <- NULL
  if(multi_rank) {
    A_perp <- ret$multi_rank$A_perp
  }
  
  sdxuv <- var_beta(X = ret$X, 
                    u = ret$u, 
                    v = ret$v, 
                    beta0 = ret$beta, 
                    d = ret$d[1], 
                    sigmay2 = ret$epsy^2,
                    sigmaa2 = ret$epsa^2, 
                    n = n, 
                    output = "uvX",
                    method = ret$method,
                    lambda = ret$l,
                    multi_rank = multi_rank,
                    multi_rank_ = ret$multi_rank,
                    A_perp = A_perp)
  
  sds <- sqrt(sdxuv)
  
  # sduv <- sqrt(rate_betauv_two_stage(X = ret$X, 
  #                                    u = ret$u, 
  #                                    v = ret$v, 
  #                                    beta0 = ret$beta, 
  #                                    d = ret$d[1], 
  #                                    sigmay2 = ret$epsy^2,
  #                                    sigmaa2 = ret$epsa^2, 
  #                                    n = n, 
  #                                    output = ret$mode) )
  # 
  # # print(rate_betauv_two_stage_check(ret$X, 
  # #                             ret$u, 
  # #                             ret$v, 
  # #                             ret$beta, 
  # #                             ret$d, 
  # #                             ret$epsa^2, 
  # #                             ret$epsy^2,
  # #                             n))
  # if(k == 0) {
  #   sdx <- NULL
  # } else {
  #   sdx <- sqrt(rate_betax_two_stage(X = ret$X, 
  #                                    u = ret$u,
  #                                    v = ret$v,
  #                                    beta0 = ret$beta,
  #                                    d = ret$d[1],
  #                                    sigmaa2 = ret$epsa^2,
  #                                    sigmay2 = ret$epsy^2,
  #                                    n = n)) 
  # }
  # sds <- c(sdx,sduv)
  
  betahat <- betahat[betahat!=0]
  
  interval <- data.frame(
    lower = betahat - sds*qnorm(1-alpha/2),
    upper = betahat + sds*qnorm(1-alpha/2)
  )
  # interval <- data.frame(
  #   lower = betahat - sds*qt(1-alpha/2, df=3),
  #   upper = betahat + sds*qt(1-alpha/2, df=3)
  # )
  
  tval <- betahat / sds
  pval <- 2*pnorm(-abs(tval))
  # pval <- 2*pt(-abs(tval), df = 3)
  
  summary_tbl <- data.frame(coef = betahat, 
                             sds = sds, 
                             t = tval, 
                             p = pval)
  # x_names <- names(sdx)
  # if(is.null(x_names)) x_names <- paste0("x_", 1:k)
  x_names <- paste0("x_", 1:k)
  uv_names <- c("u", "v")
  if(!is.null(ret$mode)) {
    if(ret$mode == "u") uv_names <- "u"
    if(ret$mode == "v") uv_names <- "v"
  }
  rownames(summary_tbl) <- c(x_names, uv_names)
  
  if(ci) {
    return(list(summary_tbl = summary_tbl, 
                interval = interval))
  } else {
    return(summary_tbl)
  }
}

#' Confidence interval and summary table of \eqn{A}
#' 
#' This function returns the confidence interval or 
#' the summary table for the adjacency matrix of 
#' \code{two_stage}, \code{supercent}
#' and \code{cv.supercent} object.
#' 
#' @param ret A  \code{two_stage}, \code{supercent}
#' and \code{cv.supercent} object
#' @param alpha The level of type-I error
#' @param ci If TRUE, return the confidence interval;
#' if FALSE, return the summary table.
#' @param A0 The true adjacency matrix
#' @return Output a data.frame of confidence interval
#' or summary table.
confint_A <- function(ret, alpha = 0.05, ci = F, A0 = NULL) {
  
  a_vec <- c(ret$d*ret$u%*%t(ret$v))
  if(!is.null(ret$A_hat)) a_vec <- c(ret$A_hat)
  
  n_train <- length(ret$u)
  
  if(grepl("two_stage", ret$method)) {
    sda <- sqrt(var_mat_A_two_stage(sigmaa2 = ret$epsa^2, 
                                    u = ret$u, v= ret$v, 
                                    n = n_train))
  }
  
  if(grepl("lr", ret$method) | ret$method == "oracle") {
    sda <- sqrt(var_mat_A_supercent(X = ret$X, l = ret$l, d = ret$d,
                                    beta0 = ret$beta,
                                    sigmay2 = ret$epsy^2,
                                    sigmaa2 = ret$epsa^2, 
                                    u = ret$u, v= ret$v, n = n_train))
  }
  
  interval <- data.frame(
    lower = a_vec - sda*qnorm(1-alpha/2),
    upper = a_vec + sda*qnorm(1-alpha/2),
    lower_FWER = a_vec - sda*qnorm(1-alpha/n_train^2/2),
    upper_FWER = a_vec + sda*qnorm(1-alpha/n_train^2/2)
  )
  
  interval$i <- rep(1:n_train, n_train)
  interval$j <- rep(1:n_train, eac = n_train)
  
  tval <- a_vec / sda
  pval <- 2*pnorm(-abs(tval))
  # pval <- 2*pt(-abs(tval), df = 3)
  
  summmary_tbl <- data.frame(coef = a_vec, 
                             sds = sda, 
                             t = tval, 
                             p = pval)
  
  if(!is.null(A0)) {
    interval$A0 <- c(A0)
    interval[, covered := (lower <= A0) & (upper >= A0)]
    interval[, covered_FWER := (lower_FWER <= A0) & (upper_FWER >= A0)]
  }
  
  if(ci) {
    return(interval)
  } else {
    return(summmary_tbl)
  }
}

#' Variance of \eqn{hat{a}^{ts}_{ij}} ordered by column major
#' 
#' @param sigmaa2 sigma_a^2
#' @param u u vector
#' @param v v vector
#' @param n number of nodes
#' @return A vector of variance of \eqn{hat{a}^{ts}_{ij}}
var_mat_A_two_stage <- function(sigmaa2, u, v, n) {
  
  I <- diag(1, nrow = n)
  Pu <- u%*%t(u)/n
  Pv <- v%*%t(v)/n
  vecuv <- c(u %*% t(v))
  
  a1 <- kron(diag((I-Pv)), diag(Pu))
  a2 <- kron(diag(Pv), diag((I-Pu)))
  a3 <- vecuv^2/n^2
  
  sigmaa2 * (a1 + a2 + a3)
  
}

#' Variance of \eqn{hat{a}_{ij}} ordered by column major
#' 
#' @param X design matrix X
#' @param l tuning parameter
#' @param sigmay2 sigma_y^2
#' @param sigmaa2 sigma_a^2
#' @param u u vector
#' @param v v vector
#' @param n number of nodes
#' @return A vector of variance of \eqn{hat{a}_{ij}}
var_mat_A_supercent <- function(X, l, d, beta0, sigmay2, sigmaa2, u, v, n) {
  
  n <- nrow(X)
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  I <- diag(1, nrow = n)
  W <- cbind(X, u, v)
  Pxuv <- W %*% solve(t(W)%*%W) %*% t(W) 
  IPxuv <- I - Pxuv
  
  Pu <- u%*%t(u)/n
  Pv <- v%*%t(v)/n
  # K <- K.mat.sparse(n, n)
  
  ## denom
  denom <- (l*d^2 + betau^2 + betav^2)
  
  ## H1 (351)
  h1a1 <- kron( diag(IPxuv), diag(Pu*n) )
  h1a2 <- kron( diag(Pv*n), diag(IPxuv) )
  h1a3 <- c(u %*% t(v) * IPxuv)
  h1a4 <- c(IPxuv * u %*% t(v))
  h1 <- betav^2 * h1a1 + betau^2 * h1a2 + betau*betav*(h1a3+h1a4)
  h1 <- h1*d^2/denom^2
  # t1_d <- diag(K %*% (kron( Pu*n, IPxuv)) %*% t(K))
  # sum((t1_d - h1a1)^2)
  # t2_d <- diag(kron( Pv*n, IPxuv))
  # sum((t2_d - h1a2)^2)
  # t3_d <- diag(K %*% (kron( u %*% t(v) , IPxuv)))
  # sum((t3_d - h1a3)^2)
  # t4_d <- diag( kron(v %*% t(u), IPxuv) %*% t(K) )
  # sum((t4_d - h1a4)^2)
  
  ## H2 (353)
  h2 <- kron( diag(I-Pv), diag(Pu) ) + kron( diag(Pv), diag(I-Pu) ) 
  
  ## H3 (354)
  h3 <- betau^4*kron( diag(Pv), diag(IPxuv) ) + betav^4*kron( diag(IPxuv), diag(Pu) ) 
  h3 <- h3/denom^2
  
  ## H4 (355)
  h4 <- kron( diag(IPxuv), diag(Pu) ) + kron( diag(Pv), diag(IPxuv) ) 
  h4 <- h4*betau^2*betav^2/denom^2
  
  ## H5 (356)
  h5 <- kron( diag(Pv), diag(Pu) )
  
  ## H23 (357)
  h23 <- betau^2*kron( diag(Pv), diag(IPxuv) ) + betav^2*kron( diag(IPxuv), diag(Pu) )
  h23 <- -h23/denom
  h32 <- h23 
  
  ## H24 (358)
  h24 <- c(u%*%t(v)/n * IPxuv) + c(IPxuv * u%*%t(v)/n)
  h24 <- -h24*betau*betav/denom
  h42 <- h24
  
  ## H34 (361)
  h34 <- betau^2*c( IPxuv * u%*%t(v)/n ) + betav^2*c( u%*%t(v)/n * IPxuv)
  h34 <- h34*betau*betav/denom^2
  
  ## H43 (362)
  h43 <- betau^2*c( u%*%t(v)/n * IPxuv ) + betav^2*c( IPxuv * u%*%t(v)/n )
  h43 <- h43*betau*betav/denom^2
  
  return(sigmay2*h1 + sigmaa2*(h2+h3+h4+h5+h23+h32+h24+h42+h34+h43))
}

# f1 <- function(l, d, beta0, sigmay2, sigmaa2, u, v, n) {
#   p <- length(beta0)
#   betau <- beta0[p-1]
#   betav <- beta0[p]
#   (2*l*d^2 + betau^2 + betav^2)/n/d^2
# }

#' Rate of centrality \eqn{\hat{u}} of SuperCENT under rank-one model
#'
#' @param l Regularization parameter
#' @param d Leading singular value
#' @param beta0 True coefficient vector, with \code{beta_u} and \code{beta_v} as last two entries
#' @param sigmay2 Variance of outcome noise
#' @param sigmaa2 Variance of network noise
#' @param n Sample size
#'
#' @return SuperCENT rate of \eqn{\hat{u}}
rate_u_lr <- function(l, d, beta0, sigmay2, sigmaa2, n) 
{
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  two_stage <- sigmaa2/d^2*(1-1/n)
  # two_stage = 0 
  s1 <- betau^2*(n-p-3)/d^2/(l*d^2 + betau^2 + betav^2)^2
  s2 <- d^2*sigmay2 - sigmaa2/n * (2*l*d^2 + betau^2 + betav^2)
  
  (two_stage + s1*s2)
}

#' Rate of \eqn{\hat{u}} of two-stage estimator under rank-one model
#'
#'
#' @param d Leading singular value
#' @param sigmaa2 Variance of network noise
#' @param n Sample size
#' @return Two-stage rate of \eqn{\hat{u}}
rate_u_two_stage <- function(d, sigmaa2, n) 
{
  sigmaa2/d^2*(1-1/n)
}

#' Optimal \eqn{\lambda} for SuperCENT
#'
#' Computes the theoretical  optimal value of \eqn{\lambda}.
#'
#' @param d Leading singular value
#' @param beta0 True coefficient vector
#' @param sigmay2 Regression noise variance
#' @param sigmaa2 Network noise variance
#' @param n Total sample size
#' @param weights Observation indicator vector
#'
#' @return Optimal regularization parameter \eqn{\lambda}
l_optimal_lr <- function(d, beta0, sigmay2, sigmaa2, n, weights = rep(1, n)) 
{
  n_obs <- sum(weights == 1)
  if(n_obs < n) print("Semi-SuperCENT")
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  n*sigmay2/sigmaa2*n/n_obs
}


#' Rate of \code{beta_u} and \code{beta_v} of SuperCENT under rank-one
#'
#' @param X Covariate matrix
#' @param u Estimated centrality vector (left)
#' @param v Estimated centrality vector (right)
#' @param beta0 True coefficient vector (with \code{beta_u}, \code{beta_v} as last entries)
#' @param d Leading singular value
#' @param sigmay2 Outcome noise variance
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#' @param l Regularization parameter
#'
#' @return Rate of \code{beta_u} and \code{beta_v} of SuperCENT 
rate_betauv_lr_2 <- function(X, u, v, beta0, d, sigmay2, sigmaa2, n, l) 
{
  if(n>100) {stop("n too big for K matrix")}
  n <- nrow(X)
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  I <- diag(1, nrow = n)
  Px <- X %*% solve(t(X)%*%X) %*% t(X)
  Pu <- u%*%t(u)/n
  Pv <- v%*%t(v)/n
  tu <- (I-Px) %*% u
  tv <- (I-Px) %*% v  
  # c <- as.numeric(t(tu)%*%tu%*%t(tv)%*%tv - (t(tu)%*%tv)^2)
  Cuv <- matrix(c(t(tu)%*%tu, t(tu)%*%tv, t(tu)%*%tv, t(tv)%*%tv), nrow = 2)
  Cuvi <- solve(Cuv)
  
  K <- K.mat.sparse(n, n)
  W <- cbind(X, u, v)
  Pxuv <- W %*% solve(t(W)%*%W) %*% t(W) 
  A1 <- I - (betau^2 + betav^2)/(l*d^2 + betau^2 + betav^2)*(I - Pxuv)
  C1 <- -1/(d*n)*(betau*t(tv)%x%(I-Pu) + betav*(t(tu)%x%(I-Pv))%*%K)
  C2 <- (betau^2 + betav^2)/(l*d^2 + betau^2 + betav^2)/d/n
  C2 <- C2 * (betau*t(tv)%x%(I-Pxuv) + betav*(t(tu)%x%(I-Pxuv))%*%K)
  
  B1 <- Cuvi %*% rbind(t(tu), t(tv)) %*% A1
  B2 <- Cuvi %*% rbind(t(tu), t(tv)) %*% (C1 + C2)
  
  ret <- sigmay2*B1 %*% t(B1) + sigmaa2*B2%*%t(B2)
  diag(ret)
}


#' Variance of \eqn{hat{\beta_u}} and \eqn{hat{\beta_v}} ordered by column major
#' 
#' @param X design matrix X
#' @param u u vector
#' @param v v vector
#' @param beta0 beta vector
#' @param d d
#' @param sigmay2 sigma_y^2
#' @param sigmaa2 sigma_a^2
#' @param n number of nodes
#' @param output "uv", "u", "v" to select output for \eqn{beta_u} and/or \eqn{beta_v}
#' @param verbose print first and second term of the rate 
#' @return A vector of rariance for \eqn{hat{\beta_u}} and \eqn{hat{\beta_v}}
var_beta <- function(X, u, v, beta0, d, sigmay2, sigmaa2, n, 
                     output = "uvX", method = "two_stage", verbose = F, 
                     lambda = NA,
                     multi_rank = F, multi_rank_ = NULL, A_perp = NULL) 
{
  
  n <- nrow(X)
  k <- ncol(X)
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  I <- diag(1, nrow = n)
  if(k == 0) {
    Px <- diag(0, nrow = n)
  } else { 
    Px <- X %*% solve(t(X)%*%X) %*% t(X)
  }
  Pu <- u%*%t(u)/n
  Pv <- v%*%t(v)/n
  tu <- (I-Px) %*% u
  tv <- (I-Px) %*% v  
  
  # (92)
  Cuv <- matrix(c(t(tu)%*%tu, t(tu)%*%tv, t(tu)%*%tv, t(tv)%*%tv), nrow = 2)
  Cuvi <- solve(Cuv)
  
  tuv <- rbind(t(tu), t(tv))
  Cuviuv <- Cuvi %*% tuv
  
  # (93)
  B1 <- Cuviuv
  B1B1t <- B1 %*% t(B1)
  
  # (97)
  AAt <- 1/(d^2*n)*(betau^2*(I-Pu) + betav^2*(I-Pv))
  B2B2t <- Cuviuv %*% AAt %*% t(Cuviuv)
  
  betax_var <- rate_betax_two_stage(X, u, v, beta0, d, sigmay2, sigmaa2, n) 
  
  # W <- cbind(X, u, v)
  # PXuv <- W %*% solve(t(W)%*%W) %*% t(W)
  # tmp1=solve(lambda*d^2*I + betav^2 *(I-PXuv))
  # tmp2 = I/(lambda*d^2 + betav^2) + betav^2/(lambda*d^2)*PXuv
  # 
  # tmp3 = (I - betav^2/(lambda*d^2 + betav^2)*(I-PXuv))/(lambda*d^2)
  # 
  # sum((tmp1 - tmp3)^2)
  
  # U_perp = multi_rank_$U_perp
  # V_perp = multi_rank_$V_perp
  # d_perp = multi_rank_$d_perp
  # 
  # K <- K.mat.sparse(n, n)
  # M11 <- t(v) %x% (I-Pu)
  # M12 <-  U_perp %*% diag(1/(1-d^2/d_perp^2)) %*% t(V_perp) %*% (t(v) %x% (I-Pu))
  # M2 <-  U_perp %*% diag(d_perp/(1-d^2/d_perp^2)) %*% t(V_perp) %*% (t(u) %x% (I-Pv) %*% K)
  # 
  
  if(multi_rank) {
    
    Cuviuv_betauv <- Cuviuv %*% cbind(-betau*I, -betav*I) 
    
    if(grepl("two_stage", method)) {
      C11C21 <- matrix(0, nrow = 2*n, ncol = n)
      C12C22 <- C12C22_two_stage(u = u, v = v, Pu = Pu, Pv = Pv, d = d, n = n, A_perp = A_perp)
      
      C31C41 <- Cuviuv
    } else {
      # SuperCENT
      C11C21_C12C22 <- C11C21_C12C22_SuperCENT(X= X, u = u, v = v, Pu = Pu, Pv = Pv, 
                                               betau = betau, betav = betav, d = d, n = n, lambda = lambda, A_perp = A_perp)
      C11C21 <- C11C21_C12C22$C11C21_eps
      C12C22 <- C11C21_C12C22$C12C22_vecE
      
      Cuviuv_betauvI <- cbind(Cuviuv_betauv, Cuviuv %*% I)
      C31C41 <- Cuviuv_betauvI %*% rbind(C11C21, I) 
      B1B1t <- C31C41 %*% t(C31C41)
    }
    
    C32C42 <- Cuviuv_betauv %*% C12C22 
    B2B2t <- C32C42 %*% t(C32C42)
    
    XtXi <- solve(t(X) %*% X)
    uv <- cbind(u, v)
    xxx <- XtXi%*%t(X)
    xxx_betauv_uv <- xxx %*% cbind(-betau*I, -betav*I, u, v) 
    
    C51 <- cbind(xxx_betauv_uv, xxx) %*% rbind(C11C21, C31C41, I)
    a1 <- sigmay2*(C51 %*% t(C51))
    
    C52 <- xxx_betauv_uv %*% rbind(C12C22, C32C42)
    a2 <- sigmaa2*(C52 %*% t(C52))
    
    betax_var <- diag(a1 + a2)
  }
  
  if(verbose) {
    print(paste0("betau First term: ", sigmay2 * B1B1t[1,1]))
    print(paste0("betau Second term: ", sigmaa2 * B2B2t[1,1]))
    
    print(paste0("betav First term: ", sigmay2 * B1B1t[2,2]))
    print(paste0("betav Second term: ", sigmaa2 * B2B2t[2,2]))
  }
  
  if(output == "uv") {
    c(sigmay2 * B1B1t[1,1] + sigmaa2 * B2B2t[1,1], sigmay2 * B1B1t[2,2] + sigmaa2 * B2B2t[2,2])
  } else if (output == "u") {
    sigmay2 * B1B1t[1,1] + sigmaa2 * B2B2t[1,1]
  } else if (output == "v") {
    sigmay2 * B1B1t[2,2] + sigmaa2 * B2B2t[2,2]
  } else if (output == "uvX") {
    c(betax_var, sigmay2 * B1B1t[1,1] + sigmaa2 * B2B2t[1,1], sigmay2 * B1B1t[2,2] + sigmaa2 * B2B2t[2,2])
  }
}


#' Variance of \eqn{hat{\beta_u}} and \eqn{hat{\beta_v}} ordered by column major
#' 
#' @param X design matrix X
#' @param u u vector
#' @param v v vector
#' @param beta0 beta vector
#' @param d d
#' @param sigmay2 sigma_y^2
#' @param sigmaa2 sigma_a^2
#' @param n number of nodes
#' @param output "uv", "u", "v" to select output for \eqn{beta_u} and/or \eqn{beta_v}
#' @param verbose print first and second term of the rate 
#' @return A vector of rariance for \eqn{hat{\beta_u}} and \eqn{hat{\beta_v}}
rate_betauv_two_stage <- function(X, u, v, beta0, d, sigmay2, sigmaa2, n, 
                                  output = "uv", verbose = F) 
{
  
  n <- nrow(X)
  k <- ncol(X)
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  I <- diag(1, nrow = n)
  if(k == 0) {
    Px <- diag(0, nrow = n)
  } else { 
    Px <- X %*% solve(t(X)%*%X) %*% t(X)
  }
  Pu <- u%*%t(u)/n
  Pv <- v%*%t(v)/n
  tu <- (I-Px) %*% u
  tv <- (I-Px) %*% v  
  
  # (92)
  Cuv <- matrix(c(t(tu)%*%tu, t(tu)%*%tv, t(tu)%*%tv, t(tv)%*%tv), nrow = 2)
  Cuvi <- solve(Cuv)
  
  tuv <- rbind(t(tu), t(tv))
  Cuviuv <- Cuvi %*% tuv
  
  # (93)
  B1 <- Cuviuv
  B1B1t <- B1 %*% t(B1)
  
  # (97)
  AAt <- 1/(d^2*n)*(betau^2*(I-Pu) + betav^2*(I-Pv))
  B2B2t <- Cuviuv %*% AAt %*% t(Cuviuv)
  
  if(verbose) {
    print(paste0("betau First term: ", sigmay2 * B1B1t[1,1]))
    print(paste0("betau Second term: ", sigmaa2 * B2B2t[1,1]))
    
    print(paste0("betav First term: ", sigmay2 * B1B1t[2,2]))
    print(paste0("betav Second term: ", sigmaa2 * B2B2t[2,2]))
  }
  
  if(output == "uv") {
    c(sigmay2 * B1B1t[1,1] + sigmaa2 * B2B2t[1,1], sigmay2 * B1B1t[2,2] + sigmaa2 * B2B2t[2,2])
  } else if (output == "u") {
    sigmay2 * B1B1t[1,1] + sigmaa2 * B2B2t[1,1]
  } else if (output == "v") {
    sigmay2 * B1B1t[2,2] + sigmaa2 * B2B2t[2,2]
  }
}

#' Compute C12 and C22 blocks for two-stage variance-covariance matrix
#'
#' @param u u vector
#' @param v v vector
#' @param Pu Projection matrix onto \code{u}, typically \code{u \%*\% t(u) / n}
#' @param Pv Projection matrix onto \code{v}
#' @param d Leading singular value
#' @param n Number of nodes
#' @param A_perp Residual adjacency matrix from multi-rank approximation (if any)
#'
#' @return A matrix representing \eqn{C_{12}} and \eqn{C_{22}} stacked vertically
#'
#' @seealso \code{\link{C11C21_C12C22_SuperCENT}}
C12C22_two_stage  <- function(u, v, Pu, Pv, d, n, A_perp) {
  
  I <- diag(1, nrow = n)
  K <- K.mat.sparse(n, n)
  
  mat1 <- rbind(cbind(n*d*I, -A_perp), cbind(t(-A_perp), n*d*I))
  
  s132 <- solve(mat1) %*% rbind(t(v)%x%(I-Pu), t(u)%x%(I-Pv)%*%K)
  
  s132
  
}

#' Compute C11, C21, C12, C22 blocks for SuperCENT variance-covariance matrix
#'
#' @param X Covariate matrix
#' @param u u vector
#' @param v v vector
#' @param Pu Projection matrix onto \code{u}
#' @param Pv Projection matrix onto \code{v}
#' @param betau Coefficient on centrality vector \code{u}
#' @param betav Coefficient on centrality vector \code{v}
#' @param d Leading singular value
#' @param n Number of nodes
#' @param lambda Regularization parameter
#' @param A_perp Residual component from multi-rank approximation (optional)
#'
#' @return A list with:
#' \describe{
#'   \item{\code{C11C21_eps}}{Influence term from response noise \eqn{\epsilon_y}}
#'   \item{\code{C12C22_vecE}}{Influence term from network noise \eqn{\epsilon_a}}
#' }
#'
#' @seealso \code{\link{C12C22_two_stage}}, \code{\link{rate_betauv_lr_2}}
C11C21_C12C22_SuperCENT  <- function(X, u, v, Pu, Pv, betau, betav, d, n, lambda, A_perp) {
  
  I <- diag(1, nrow = n)
  K <- K.mat.sparse(n, n)
  
  mat0 <- lambda*d^2*diag(1, nrow = 2*n)
  mat1 <- rbind(cbind(n*d*I, -A_perp), cbind(t(-A_perp), n*d*I))
  
  W <- cbind(X, u, v)
  PXuv <- W %*% solve(t(W)%*%W) %*% t(W)
  mat2 <- rbind(c(betau^2, betau*betav), c(betau*betav, betav^2)) %x% (I - PXuv)
  
  eps <- solve(lambda*d/n*mat1 + mat2) %*% rbind(betau*(I-PXuv), betav*(I-PXuv))
  
  vecE <- solve(lambda*d/n*mat1 + mat2) %*% (lambda*d/n*rbind(t(v)%x%(I-Pu), t(u)%x%(I-Pv)%*%K))
  
  return(list(C11C21_eps = eps, 
              C12C22_vecE = vecE))
}

#' Debugging check for variance of \eqn{\\hat{\\beta}_u} in two-stage estimator
#'
#' @param X Covariate matrix
#' @param u u vector
#' @param v v vector
#' @param beta0 True coefficient vector
#' @param d Leading singular value
#' @param sigmay2 Outcome noise variance
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#'
#' @return A list of intermediate terms including total rate, signal and network variance terms, and key scalars
rate_betauv_two_stage_check <- function(X, u, v, beta0, d, sigmay2, sigmaa2, n) 
{
  # check (305) = (311) = (312)
  
  n <- nrow(X)
  k <- ncol(X)
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  I <- diag(1, nrow = n)
  if(k == 0) {
    Px <- diag(0, nrow = n)
  } else { 
    Px <- X %*% solve(t(X)%*%X) %*% t(X)
  }
  
  I <- diag(1, nrow = n)
  Pu <- u%*%t(u)/n
  Pv <- v%*%t(v)/n
  tu <- (I-Px) %*% u
  tv <- (I-Px) %*% v  
  c <- as.numeric(t(tu)%*%tu%*%t(tv)%*%tv - (t(tu)%*%tv)^2)
  tvtv <- as.numeric(t(tv)%*%tv)
  
  a1 <- sigmay2/c*tvtv
  
  b1 <- as.numeric(t(tv)%*%tv%*%t(tu)%*%(I-Pv)%*%tu%*%t(tv)%*%tv)
  b2 <- as.numeric(t(tu)%*%tv%*%t(tv)%*%(I-Pu)%*%tv%*%t(tu)%*%tv)
  a2 <- sigmaa2/c^2/d^2/n*(betav^2*b1 + betau^2*b2)
  
  list(betau_rate0 = a1 + a2,
       a1 = a1, 
       a2 = a2,
       sigmay2 = sigmay2,
       sigmaa2 = sigmaa2,
       c = c,
       tvtv = tvtv,
       b1 = b1,
       b2 = b2,
       d = d
  )
  
}

#' Asymptotic variance of \eqn{\\hat{\\beta}_x} under two-stage 
#'
#' @param X Covariate matrix
#' @param u u vector
#' @param v v vector
#' @param beta0 True coefficient vector
#' @param d Leading singular value
#' @param sigmay2 Outcome noise variance
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#'
#' @return A vector of variances for each component of \eqn{\\hat{\\beta}_x}
#'
#' @seealso \code{\link{rate_betax_lr}}
rate_betax_two_stage <- function(X, u, v, beta0, d, sigmay2, sigmaa2, n) 
{
  n <- nrow(X)
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  I <- diag(1, nrow = n)
  Px <- X %*% solve(t(X)%*%X) %*% t(X)
  Pu <- u%*%t(u)/n
  Pv <- v%*%t(v)/n
  tu <- (I-Px) %*% u
  tv <- (I-Px) %*% v  
  c <- as.numeric(t(tu)%*%tu%*%t(tv)%*%tv - (t(tu)%*%tv)^2)
  Cuv <- matrix(c(t(tu)%*%tu, t(tu)%*%tv, t(tu)%*%tv, t(tv)%*%tv), nrow = 2)
  Cuvi <- solve(Cuv)
  
  XtXi <- solve(t(X) %*% X)
  uv <- cbind(u, v)
  xxx <- XtXi%*%t(X)
  
  a1 <- sigmay2*(XtXi + xxx%*%uv%*%Cuvi%*%t(uv)%*%t(xxx) )
  
  b1 <- betau^2*(I-Pu) + betav^2*(I-Pv)
  b2 <- matrix(c(betav^2*t(tu)%*%(I-Pv)%*%tu, 0, 0, betau^2*t(tv)%*%(I-Pu)%*%tv), 2)
  b3 <- uv %*% Cuvi %*% b2 %*% Cuvi %*% t(uv)
  a2 <- sigmaa2/d^2/n*(xxx%*%(b1 + b3)%*%t(xxx))
  
  diag(a1 + a2)
}

#' Asymptotic variance of \eqn{\\hat{\\beta}_x} under SuperCENT estimator
#'
#' @param X Covariate matrix
#' @param u u vector
#' @param v v vector
#' @param beta0 True coefficient vector
#' @param d Leading singular value
#' @param sigmay2 Outcome noise variance
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#'
#' @return A vector of variances for each component of \eqn{\\hat{\\beta}_x}
#'
#' @seealso \code{\link{rate_betax_two_stage}}
rate_betax_lr <- function(X, u, v, beta0, d, sigmay2, sigmaa2, n) 
{
  n <- nrow(X)
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  I <- diag(1, nrow = n)
  Px <- X %*% solve(t(X)%*%X) %*% t(X)
  Pu <- u%*%t(u)/n
  Pv <- v%*%t(v)/n
  tu <- (I-Px) %*% u
  tv <- (I-Px) %*% v  
  c <- as.numeric(t(tu)%*%tu%*%t(tv)%*%tv - (t(tu)%*%tv)^2)
  Cuv <- matrix(c(t(tu)%*%tu, t(tu)%*%tv, t(tu)%*%tv, t(tv)%*%tv), nrow = 2)
  Cuvi <- solve(Cuv)
  
  XtXi <- solve(t(X) %*% X)
  uv <- cbind(u, v)
  xxx <- XtXi%*%t(X)
  
  a1 <- sigmay2*(XtXi + xxx%*%uv%*%Cuvi%*%t(uv)%*%t(xxx) )
  
  b1 <- betau^2*(I-Pu) + betav^2*(I-Pv)
  b2 <- matrix(c(betav^2*t(tu)%*%(I-Pv)%*%tu, 0, 0, betau^2*t(tv)%*%(I-Pu)%*%tv), 2)
  b3 <- t(uv) %*% Cuvi %*% b2 %*% Cuvi %*% uv
  a2 <- sigmaa2/d^2/n*(xxx%*%(b1 + b3)%*%t(xxx))
  
  diag(a1 + a2)
}


#' Helper: Common penalty term in SuperCENT risk expression
#'
#' @param l Regularization parameter
#' @param d Leading singular value
#' @param betau Coefficient on \code{u}
#' @param betav Coefficient on \code{v}
#' @param sigmay2 Outcome noise variance
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#'
#' @return The combined penalty term used in risk difference expressions
func <- function(l, d, betau, betav, sigmay2, sigmaa2, n) 
{
  common <- (betau^2 + betav^2)*(2*l*d^2 + betau^2 + betav^2)/(l*d^2 + betau^2 + betav^2)^2
  a1 <- -sigmay2*common
  a2 <- -sigmaa2*common*(betau^2 + betav^2)/d^2/n
  
  a1+a2
}

#' Asymptotic prediction risk under two-stage network regression
#'
#' @param beta0 True coefficient vector
#' @param d Leading singular value
#' @param sigmay2 Outcome noise variance
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#'
#' @return Risk value
rate_risk_two_stage <- function(beta0, d, sigmay2, sigmaa2, n)
{
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  (n-p-3) * (sigmay2 + sigmaa2/(d^2*n)*(betau^2 + betav^2))
}

#' Asymptotic prediction risk under SuperCENT
#'
#' @param l Regularization parameter
#' @param u u vector
#' @param v v vector
#' @param beta0 True coefficient vector
#' @param d Leading singular value
#' @param sigmay2 Outcome noise variance
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#'
#' @return Scalar risk value for SuperCENT
#'
#' @seealso \code{\link{rate_risk_two_stage}}
rate_risk_lr <- function(l, u, v, beta0, d, sigmay2, sigmaa2, n)
{
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  first_multiplier <- (n-p-3)*(betau^2 + betav^2)*(2*l*d^2 + betau^2 + betav^2)/(l*d^2 + betau^2 + betav^2)^2
  second_multiplier <- sigmay2 + sigmaa2/(d^2*n)
  
  rate_risk_two_stage(beta0, d, sigmay2, sigmaa2, n) - first_multiplier*second_multiplier
}

#' Expected squared error of adjacency matrix recovery under two-stage
#'
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#'
#' @return Square error
rate_A_two_stage <- function(sigmaa2, n)
{
  sigmaa2 * (2*n - 1)
}

#' Expected squared error of adjacency matrix recovery under SuperCENT
#'
#' @param l Regularization parameter
#' @param u u vector
#' @param v v vector
#' @param beta0 True coefficient vector
#' @param d Leading singular value
#' @param sigmay2 Outcome noise variance
#' @param sigmaa2 Network noise variance
#' @param n Sample size
#'
#' @return Square error
#'
#' @seealso \code{\link{rate_A_two_stage}}
rate_A_lr <- function(l, u, v, beta0, d, sigmay2, sigmaa2, n) 
{
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  common_multiplier <- (n-p-3)/(l*d^2 + betau^2 + betav^2)^2
  sec_term <- common_multiplier*(betau^2 + betav^2)*(n*d^2*sigmay2 - (2*l*d^2 + betau^2 + betav^2)*sigmaa2)
  third_term <- 2*sigmaa2*common_multiplier*(1 + (t(v)%*%u/n)^2 - betau^2*betav^2)
  
  first_term <- rate_A_two_stage(sigmaa2, n)
  
  return(first_term + sec_term + third_term)
}


#' Experimental: Subsample-based confidence intervals for network regression
#'
#' Implements HULC confidence intervals (Kuchibhotla et al., 2021)
#' for regression coefficients in network regression models. The procedure resamples
#' blocks from the training set and refits the model multiple times using a user-defined estimator.
#'
#' @param A_train Adjacency matrix for training nodes
#' @param X_train Covariate matrix for training nodes
#' @param y_train Response vector for training nodes
#' @param B Number of subsample blocks (default \code{5})
#' @param FUN Estimator function (e.g., \code{supercent}) that returns a list with element \code{beta}
#' @param ... Additional arguments passed to \code{FUN} (e.g., \code{l}, \code{lrange}, \code{gap}, \code{folds}, \code{max_iter})
#'
#' @return A matrix of lower and upper bounds for each coefficient in \code{beta}, including centrality effects
#'
#' @details
#' Each subsample block fits a model on a subset of the training data and extracts the estimated coefficients.
#' The final interval is formed by taking the componentwise minimum and maximum across all subsample fits.
#'
#' @note Assumes \code{beta0vec} is available in the environment to align signs via \code{beta_sign()}.
#'
#' @seealso \code{\link{beta_sign}}, \code{\link{confint_all}}
#'
#' @examples
#' # hulc(A_train, X_train, y_train, FUN = supercent, B = 5, l = 0.1)
hulc <- function(A_train, X_train, y_train, B = 5, FUN, ...) {
  
  input_list <- list(...)
  l <- lrange <- gap <- folds <- max_iter <- NULL
  if(!is.null(input_list$l)) l <- input_list$l
  if(!is.null(input_list$lrange)) lrange <- input_list$lrange
  if(!is.null(input_list$gap)) gap <- input_list$gap
  if(!is.null(input_list$folds)) folds <- input_list$folds
  if(!is.null(input_list$max_iter)) max_iter <- input_list$max_iter
  
  set.seed(1)
  idx <- seq_along(y_train)
  idx_s <- sample(idx, replace = F)
  idx_list <- split(idx_s, ceiling(idx/length(idx)*B))
  
  beta_list <- lapply(idx_list,
                      function(i) FUN(A_train[i, i], X_train[i, ], y_train[i], 
                                      l = l, lrange = lrange, gap = gap, 
                                      folds = folds, max_iter = max_iter)$beta)
  beta_list <- lapply(beta_list, function(beta) beta_sign(beta, beta0vec))
  
  beta_mat <- matrix(unlist(beta_list), ncol = ncol(X_train)+2, byrow = T)
  interval <- cbind(lower = apply(beta_mat, 2, min),
                    upper = apply(beta_mat, 2, max))
  
  interval
}
