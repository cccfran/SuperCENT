library(Matrix)

vec_norm <- function(x) {sqrt(sum(x^2))}
spec_norm_diff <- function(x, y, scale = T) {
  if(length(x) != length(y)) stop('x and y are of different length')
  if(length(x) != 1 & scale) {
    x <- x/sqrt(sum(x^2))
    y <- y/sqrt(sum(y^2))
  }
  norm(x %*% t(x) - y %*% t(y), "2")
}

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

two_stage <- function(A, X, y, r = 1, scaled = 1, weights = rep(1, length(y)), ...) {
  
  n = nrow(A)
  k = ifelse(is.null(X), 0, ncol(X))
  n_obs = sum(weights == 1)
  
  # move the unobserved to the end
  A_ori <- A
  if(n_obs < n) A <- A_shift_unobserved_last(A_ori, weights)
  
  # SVD
  svda <- irlba::irlba(A, nu = r, nv = r)
  uu <- svda$u[,1]
  vv <- svda$v[,1]
  d <- svda$d[1]
  
  # OLS
  ret_two_stage <- lm.fit(cbind(X, uu[1:n_obs], vv[1:n_obs]), y)
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
  
  ret_two_stage$A <- A_ori
  ret_two_stage$X <- X
  ret_two_stage$y <- y
  
  ret_two_stage$epsa <- epsa_hat(ret_two_stage)
  ret_two_stage$epsy <- epsy_hat(ret_two_stage)
  
  ret_two_stage$iter <- NA
  ret_two_stage$l <- NA
  
  class(ret_two_stage) <- "two_stage"
  
  ret_two_stage
}

lopt_estimate <- function(A, X, y, weights) {
  
  n <- length(y)
  n_obs <- sum(weights == 1)
  
  ret_two_stage <- two_stage(A, X, y, weights = weights)
  sigmayhat2 <- epsy_hat(ret_two_stage)^2
  sigmaahat2 <- epsa_hat(ret_two_stage)^2
  
  l <- n*sigmayhat2/sigmaahat2*n/n_obs
  
  l
}

supercent <- function(A, X, y, l = NULL, tol = 1e-4, max_iter = 200, 
                      weights = rep(1, length(y)), verbose = 0, ...) {
  
  n = nrow(A)
  k = ifelse(is.null(X), 0, ncol(X))
  n_obs = sum(weights == 1)
  
  # move the unobserved to the end
  A_ori <- A
  if(n_obs < n) {
    print("Semi-SuperCENT")
    A <- A_shift_unobserved_last(A_ori, weights)
  }
  
  if(is.null(l)) l <- lopt_estimate(A, X, y, weights)
  
  ret <- lr(A, X, y, l, tol, max_iter, verbose)
  
  # Adjust sign
  ret <- adjust_sign(ret)
  
  # shift the order back
  if(n_obs < n) {
    ret$u <- unobserved_shift_back(ret$u, weights)
    ret$v <- unobserved_shift_back(ret$v, weights)
  }
  
  ret$A <- A_ori
  ret$X <- X
  ret$y <- y
  
  ret$epsa <- epsa_hat(ret)
  ret$epsy <- epsy_hat(ret)
  
  class(ret) <- "supercent"
  
  ret
}


cv.supercent <- function(A, X, y, 
                         l = NULL, lrange = 2^4, gap = 2,
                         folds = 10, tol = 1e-4, max_iter = 200, 
                         weights = rep(1, length(y)), verbose = 0, ...) {
  
  n = nrow(A)
  k = ifelse(is.null(X), 0, ncol(X))
  n_obs = sum(weights == 1)
  
  # move the unobserved to the end
  A_ori <- A
  if(n_obs < n) A <- A_shift_unobserved_last(A_ori, weights)
  
  if(is.null(l)) l <- lopt_estimate(A, X, y, weights)
  
  lmin <- l/lrange*(2^gap)
  # lmin <- l
  lmax <- l*lrange
  
  if(n_obs < n) {
    print("Semi-SuperCENT")
    ret <- cv_lr_2(A, X, y, lmin, lmax, gap, tol, max_iter, folds, verbose) 
  } else {
    print("SuperCENT")
    ret <- cv_lr(A, X, y, lmin, lmax, gap, tol, max_iter, folds, verbose) 
  }
  
  # Adjust sign
  ret <- adjust_sign(ret)
  
  # shift the order back
  if(n_obs < n) {
    ret$u <- unobserved_shift_back(ret$u, weights)
    ret$v <- unobserved_shift_back(ret$v, weights)
  }
  
  ret$A <- A_ori
  ret$X <- X
  ret$y <- y
  
  ret$epsa <- epsa_hat(ret)
  ret$epsy <- epsy_hat(ret)
  
  class(ret) <- "cv.supercent"
  
  ret
}

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

oracle <- function(A, X, y, d, U, V, beta0vec, beta_hat, epsa, epsy,
                   A_hat, l = NA,
                   weights = observed, method) {
  
  unobs <- which(!weights)
  obs <- which(weights == 1)
  
  ret <- ret_constructor(A, X, y)
  
  ret$d <- d
  ret$u <- U
  ret$v <- V
  ret$beta <- beta0vec
  ret$beta_hat <- beta_hat
  
  ret$residuals <- (cbind(X, U[obs,], V[obs,]) %*% beta0vec - y)
  
  df <- opt$n_train - length(beta0vec)
  
  ret$A <- A
  ret$A_hat <- A_hat
  ret$X <- X
  ret$y <- y
  
  ret$epsa <- epsa
  ret$epsy <- epsy
  
  ret$l <- l
  
  ret$method <- method
  
  ret
}

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

# redundant
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
  
  cbind(X, u_test, v_test) %*% ret$beta

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

beta_sign <- function(beta, beta0)
{
  p <- length(beta0)
  beta[p-1] <- ifelse(beta[p-1]*beta0[p-1] < 0, -beta[p-1], beta[p-1])
  beta[p] <- ifelse(beta[p]*beta0[p] < 0, -beta[p], beta[p])
  
  beta
}

u_sign <- function(u, u0)
{
  sgn <- sign(c(t(u0)%*%u))
  u <- sgn*u
  
  u
}

ret_sign <- function(ret, beta0vec, u0, v0)
{
  if(!is.null(ret$beta_hat)) ret$beta_hat <- beta_sign(ret$beta_hat, beta0vec)
  ret$beta <- beta <- beta_sign(ret$beta, beta0vec)
  ret$u <- u_sign(ret$u, u0)
  ret$v <- u_sign(ret$v, v0)
  
  ret
}

#' Calculate variance
#' 
#' @param X_train X train matrix
#' @param method method
#' @return variance of hat u,v,betax, betau, betav
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

epsa_hat_oracle <- function(A0, d, u, v) {
  n <- length(u)
  sqrt(sum((d * u%*%t(v) - A0)^2)/(n^2-1))
}

epsa_hat <- function(ret) {
  n <- length(ret$u)
  if(!is.null(ret$v)) {
    epsa2 <- sum((ret$d*ret$u%*%t(ret$v) - ret$A)^2)/n^2
  } else {
    epsa2 <- sum((ret$d*ret$u%*%t(ret$u) - ret$A)^2)/n^2
  }
  
  sqrt(epsa2)
}

epsy_hat <- function(ret) {
  n <- length(ret$residuals)
  p <- length(ret$beta)
  
  sqrt(sum(ret$residuals^2)/(n - p))
}

confint <- function(ret, alpha = 0.05, ci = F) {
  
  n <- length(ret$y)
  k <- ncol(ret$X)
  betahat <- ret$beta
  # for oracle
  if(!is.null(ret$beta_hat)) betahat <- ret$beta_hat

  sduv <- sqrt(rate_betauv_two_stage(X = ret$X, 
                                     u = ret$u, 
                                     v = ret$v, 
                                     beta0 = ret$beta, 
                                     d = ret$d, 
                                     sigmay2 = ret$epsy^2,
                                     sigmaa2 = ret$epsa^2, 
                                     n = n, 
                                     output = "uv") )
  
  # print(rate_betauv_two_stage_check(ret$X, 
  #                             ret$u, 
  #                             ret$v, 
  #                             ret$beta, 
  #                             ret$d, 
  #                             ret$epsa^2, 
  #                             ret$epsy^2,
  #                             n))
  if(k == 0) {
    sdx <- NULL
  } else {
    sdx <- sqrt(rate_betax_two_stage(X = ret$X, 
                                     u = ret$u,
                                     v = ret$v,
                                     beta0 = ret$beta,
                                     d = ret$d,
                                     sigmaa2 = ret$epsa^2,
                                     sigmay2 = ret$epsy^2,
                                     n = n)) 
  }
  sds <- c(sdx,sduv)
  
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
  
  summmary_tbl <- data.frame(coef = betahat, 
                             sds = sds, 
                             t = tval, 
                             p = pval)
  x_names <- names(sdx)
  if(is.null(x_names)) x_names <- paste0("x_", 1:k)
  rownames(summmary_tbl) <- c(x_names, "u", "v")
  
  if(ci) {
    return(interval)
  } else {
    return(summmary_tbl)
  }
}

confint_A <- function(ret, ci = F, A0 = NULL) {
  
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
  
  interval <- data.table(
    lower = a_vec - sda*qnorm(1-alpha/2),
    upper = a_vec + sda*qnorm(1-alpha/2),
    lower_FWER = a_vec - sda*qnorm(1-alpha/n_train^2/2),
    upper_FWER = a_vec + sda*qnorm(1-alpha/n_train^2/2)
  )

  interval[, i := rep(1:n_train, n_train)]
  interval[, j := rep(1:n_train, eac = n_train)]
  
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

#' Variance of $\hat{a}^{ts}_{ij}$ ordered by column major
#' 
#' @param sigmaa2 sigma_a^2
#' @param u u vector
#' @param v v vector
#' @param n number of nodes
#' @return A vector of variance of $\hat{a}^{ts}_{ij}$
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

#' Variance of $\hat{a}_{ij}$ ordered by column major
#' 
#' @param X design matrix X
#' @param l tuning parameter
#' @param sigmay2 sigma_y^2
#' @param sigmaa2 sigma_a^2
#' @param u u vector
#' @param v v vector
#' @param n number of nodes
#' @return A vector of variance of $\hat{a}_{ij}$
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

f1 <- function(l, d, beta0, sigmay2, sigmaa2, u, v, n) {
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  (2*l*d^2 + betau^2 + betav^2)/n/d^2
}

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

rate_u_two_stage <- function(d, sigmaa2, n) 
{
  sigmaa2/d^2*(1-1/n)
}

l_optimal_lr <- function(d, beta0, sigmay2, sigmaa2, n, weights = rep(1, n)) 
{
  n_obs <- sum(weights == 1)
  if(n_obs < n) print("Semi-SuperCENT")
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]

  n*sigmay2/sigmaa2*n/n_obs
}

rate_betau_two_stage <- function(X, u, v, beta0, d, sigmay2, sigmaa2, n) 
{
  print("there is an error. Do NOT use")
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
  
  a1 <- sigmay2/c*as.numeric(t(tv)%*%tv)
  
  b1 <- as.numeric(betav^2*t(tv)%*%tv%*%t(tu)%*%(I-Pv)%*%tu%*%t(tv)%*%tv)
  b2 <- as.numeric(betau^2*t(tu)%*%tv%*%t(tv)%*%(I-Pu)%*%tv%*%t(tu)%*%tv)
  a2 <- sigmaa2/c^2/d^2/n*(b1 + b2)
  
  a1 + a2
}


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

#' Variance of $\hat{\beta_u}$ and $\hat{\beta_v}$ ordered by column major
#' 
#' @param X design matrix X
#' @param u u vector
#' @param v v vector
#' @param beta0 beta vector
#' @param d d
#' @param sigmay2 sigma_y^2
#' @param sigmaa2 sigma_a^2
#' @param n number of nodes
#' @param output "uv", "u", "v" to select output for $\beta_u$ and/or $\beta_v$
#' @param verbose print first and second term of the rate 
#' @return A vector of rariance for $\hat{\beta_u}$ and $\hat{\beta_v}$
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

func <- function(l, d, betau, betav, sigmay2, sigmaa2, n) 
{
  common <- (betau^2 + betav^2)*(2*l*d^2 + betau^2 + betav^2)/(l*d^2 + betau^2 + betav^2)^2
  a1 <- -sigmay2*common
  a2 <- -sigmaa2*common*(betau^2 + betav^2)/d^2/n
  
  a1+a2
}

rate_risk_two_stage <- function(beta0, d, sigmay2, sigmaa2, n)
{
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  (n-p-3) * (sigmay2 + sigmaa2/(d^2*n)*(betau^2 + betav^2))
}

rate_risk_lr <- function(l, u, v, beta0, d, sigmay2, sigmaa2, n)
{
  p <- length(beta0)
  betau <- beta0[p-1]
  betav <- beta0[p]
  
  first_multiplier <- (n-p-3)*(betau^2 + betav^2)*(2*l*d^2 + betau^2 + betav^2)/(l*d^2 + betau^2 + betav^2)^2
  second_multiplier <- sigmay2 + sigmaa2/(d^2*n)
  
  rate_risk_two_stage(beta0, d, sigmay2, sigmaa2, n) - first_multiplier*second_multiplier
}

rate_A_two_stage <- function(sigmaa2, n)
{
  sigmaa2 * (2*n - 1)
}

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
