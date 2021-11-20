// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _SuperCENT_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _SuperCENT_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _SuperCENT_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _SuperCENT_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}
// cent
Rcpp::List cent(const arma::mat& A, const arma::mat& X, const arma::colvec& y, double l1, double l2, double tol, int max_iter, int verbose, int scaled);
RcppExport SEXP _SuperCENT_cent(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP verboseSEXP, SEXP scaledSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type scaled(scaledSEXP);
    rcpp_result_gen = Rcpp::wrap(cent(A, X, y, l1, l2, tol, max_iter, verbose, scaled));
    return rcpp_result_gen;
END_RCPP
}
// cv_oracle_cent
Rcpp::List cv_oracle_cent(const arma::mat& A, const arma::mat& X, const arma::colvec& y, double lmin, double lmax, double gap, double tol, int max_iter, const arma::colvec& beta0, int verbose, int scaled, int scaledn, Nullable<NumericVector> d0_, Nullable<NumericVector> u0_, Nullable<NumericVector> v0_, Nullable<NumericMatrix> X_test_, Nullable<NumericVector> y_test_, Nullable<NumericVector> u_test0_, Nullable<NumericVector> v_test0_);
RcppExport SEXP _SuperCENT_cv_oracle_cent(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP lminSEXP, SEXP lmaxSEXP, SEXP gapSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP beta0SEXP, SEXP verboseSEXP, SEXP scaledSEXP, SEXP scalednSEXP, SEXP d0_SEXP, SEXP u0_SEXP, SEXP v0_SEXP, SEXP X_test_SEXP, SEXP y_test_SEXP, SEXP u_test0_SEXP, SEXP v_test0_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lmin(lminSEXP);
    Rcpp::traits::input_parameter< double >::type lmax(lmaxSEXP);
    Rcpp::traits::input_parameter< double >::type gap(gapSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type scaled(scaledSEXP);
    Rcpp::traits::input_parameter< int >::type scaledn(scalednSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d0_(d0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type u0_(u0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type v0_(v0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type X_test_(X_test_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type y_test_(y_test_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type u_test0_(u_test0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type v_test0_(v_test0_SEXP);
    rcpp_result_gen = Rcpp::wrap(cv_oracle_cent(A, X, y, lmin, lmax, gap, tol, max_iter, beta0, verbose, scaled, scaledn, d0_, u0_, v0_, X_test_, y_test_, u_test0_, v_test0_));
    return rcpp_result_gen;
END_RCPP
}
// cv_cent
Rcpp::List cv_cent(const arma::mat& A, const arma::mat& X, const arma::colvec& y, double lmin, double lmax, double gap, double tol, int max_iter, int folds, int verbose, int scaled, int scaledn, Nullable<NumericVector> d0_, Nullable<NumericVector> u0_, Nullable<NumericVector> v0_);
RcppExport SEXP _SuperCENT_cv_cent(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP lminSEXP, SEXP lmaxSEXP, SEXP gapSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP foldsSEXP, SEXP verboseSEXP, SEXP scaledSEXP, SEXP scalednSEXP, SEXP d0_SEXP, SEXP u0_SEXP, SEXP v0_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lmin(lminSEXP);
    Rcpp::traits::input_parameter< double >::type lmax(lmaxSEXP);
    Rcpp::traits::input_parameter< double >::type gap(gapSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type folds(foldsSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type scaled(scaledSEXP);
    Rcpp::traits::input_parameter< int >::type scaledn(scalednSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d0_(d0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type u0_(u0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type v0_(v0_SEXP);
    rcpp_result_gen = Rcpp::wrap(cv_cent(A, X, y, lmin, lmax, gap, tol, max_iter, folds, verbose, scaled, scaledn, d0_, u0_, v0_));
    return rcpp_result_gen;
END_RCPP
}
// lr
Rcpp::List lr(const arma::mat& A, const arma::mat& X, const arma::colvec& y, double l, double tol, int max_iter, int verbose, int scaled);
RcppExport SEXP _SuperCENT_lr(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP lSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP verboseSEXP, SEXP scaledSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type l(lSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type scaled(scaledSEXP);
    rcpp_result_gen = Rcpp::wrap(lr(A, X, y, l, tol, max_iter, verbose, scaled));
    return rcpp_result_gen;
END_RCPP
}
// cv_oracle_lr
Rcpp::List cv_oracle_lr(const arma::mat& A, const arma::mat& X, const arma::colvec& y, double lmin, double lmax, double gap, double tol, int max_iter, const arma::colvec& beta0, int verbose, int scaled, int scaledn, Nullable<NumericVector> d0_, Nullable<NumericVector> u0_, Nullable<NumericVector> v0_, Nullable<NumericMatrix> X_test_, Nullable<NumericVector> y_test_, Nullable<NumericVector> u_test0_, Nullable<NumericVector> v_test0_);
RcppExport SEXP _SuperCENT_cv_oracle_lr(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP lminSEXP, SEXP lmaxSEXP, SEXP gapSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP beta0SEXP, SEXP verboseSEXP, SEXP scaledSEXP, SEXP scalednSEXP, SEXP d0_SEXP, SEXP u0_SEXP, SEXP v0_SEXP, SEXP X_test_SEXP, SEXP y_test_SEXP, SEXP u_test0_SEXP, SEXP v_test0_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lmin(lminSEXP);
    Rcpp::traits::input_parameter< double >::type lmax(lmaxSEXP);
    Rcpp::traits::input_parameter< double >::type gap(gapSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type beta0(beta0SEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type scaled(scaledSEXP);
    Rcpp::traits::input_parameter< int >::type scaledn(scalednSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d0_(d0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type u0_(u0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type v0_(v0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericMatrix> >::type X_test_(X_test_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type y_test_(y_test_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type u_test0_(u_test0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type v_test0_(v_test0_SEXP);
    rcpp_result_gen = Rcpp::wrap(cv_oracle_lr(A, X, y, lmin, lmax, gap, tol, max_iter, beta0, verbose, scaled, scaledn, d0_, u0_, v0_, X_test_, y_test_, u_test0_, v_test0_));
    return rcpp_result_gen;
END_RCPP
}
// cv_lr
Rcpp::List cv_lr(const arma::mat& A, const arma::mat& X, const arma::colvec& y, double lmin, double lmax, double gap, double tol, int max_iter, int folds, int verbose, int scaled, int scaledn, Nullable<NumericVector> d0_, Nullable<NumericVector> u0_, Nullable<NumericVector> v0_);
RcppExport SEXP _SuperCENT_cv_lr(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP lminSEXP, SEXP lmaxSEXP, SEXP gapSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP foldsSEXP, SEXP verboseSEXP, SEXP scaledSEXP, SEXP scalednSEXP, SEXP d0_SEXP, SEXP u0_SEXP, SEXP v0_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lmin(lminSEXP);
    Rcpp::traits::input_parameter< double >::type lmax(lmaxSEXP);
    Rcpp::traits::input_parameter< double >::type gap(gapSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type folds(foldsSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type scaled(scaledSEXP);
    Rcpp::traits::input_parameter< int >::type scaledn(scalednSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d0_(d0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type u0_(u0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type v0_(v0_SEXP);
    rcpp_result_gen = Rcpp::wrap(cv_lr(A, X, y, lmin, lmax, gap, tol, max_iter, folds, verbose, scaled, scaledn, d0_, u0_, v0_));
    return rcpp_result_gen;
END_RCPP
}
// cv_lr_2
Rcpp::List cv_lr_2(const arma::mat& A, const arma::mat& X, const arma::colvec& y, double lmin, double lmax, double gap, double tol, int max_iter, int folds, int verbose, int scaled, int scaledn, Nullable<NumericVector> d0_, Nullable<NumericVector> u0_, Nullable<NumericVector> v0_);
RcppExport SEXP _SuperCENT_cv_lr_2(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP lminSEXP, SEXP lmaxSEXP, SEXP gapSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP foldsSEXP, SEXP verboseSEXP, SEXP scaledSEXP, SEXP scalednSEXP, SEXP d0_SEXP, SEXP u0_SEXP, SEXP v0_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type lmin(lminSEXP);
    Rcpp::traits::input_parameter< double >::type lmax(lmaxSEXP);
    Rcpp::traits::input_parameter< double >::type gap(gapSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type folds(foldsSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type scaled(scaledSEXP);
    Rcpp::traits::input_parameter< int >::type scaledn(scalednSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type d0_(d0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type u0_(u0_SEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type v0_(v0_SEXP);
    rcpp_result_gen = Rcpp::wrap(cv_lr_2(A, X, y, lmin, lmax, gap, tol, max_iter, folds, verbose, scaled, scaledn, d0_, u0_, v0_));
    return rcpp_result_gen;
END_RCPP
}
// predict
Rcpp::List predict(const arma::mat& A, const arma::colvec& u_train, const arma::colvec& v_train, const arma::colvec& beta, const arma::mat& X_test, double tol, int max_iter);
RcppExport SEXP _SuperCENT_predict(SEXP ASEXP, SEXP u_trainSEXP, SEXP v_trainSEXP, SEXP betaSEXP, SEXP X_testSEXP, SEXP tolSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type u_train(u_trainSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type v_train(v_trainSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(predict(A, u_train, v_train, beta, X_test, tol, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// predict_svd
Rcpp::List predict_svd(const arma::mat& A, const arma::colvec& beta, const arma::mat& X_test);
RcppExport SEXP _SuperCENT_predict_svd(SEXP ASEXP, SEXP betaSEXP, SEXP X_testSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    rcpp_result_gen = Rcpp::wrap(predict_svd(A, beta, X_test));
    return rcpp_result_gen;
END_RCPP
}
// predict_oracle
Rcpp::List predict_oracle(const arma::mat& A, const arma::colvec& beta, const arma::mat& X_test, const arma::colvec& u_test, const arma::colvec& v_test);
RcppExport SEXP _SuperCENT_predict_oracle(SEXP ASEXP, SEXP betaSEXP, SEXP X_testSEXP, SEXP u_testSEXP, SEXP v_testSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type u_test(u_testSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type v_test(v_testSEXP);
    rcpp_result_gen = Rcpp::wrap(predict_oracle(A, beta, X_test, u_test, v_test));
    return rcpp_result_gen;
END_RCPP
}
// lr_check
Rcpp::List lr_check(const arma::mat& A, const arma::mat& X, const arma::colvec& y, arma::colvec& u, arma::colvec& v, arma::colvec& beta, double d, double l, double tol, int max_iter, int verbose, bool uv_is_init, arma::colvec& u0);
RcppExport SEXP _SuperCENT_lr_check(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP uSEXP, SEXP vSEXP, SEXP betaSEXP, SEXP dSEXP, SEXP lSEXP, SEXP tolSEXP, SEXP max_iterSEXP, SEXP verboseSEXP, SEXP uv_is_initSEXP, SEXP u0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type u(uSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type v(vSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type l(lSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type uv_is_init(uv_is_initSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type u0(u0SEXP);
    rcpp_result_gen = Rcpp::wrap(lr_check(A, X, y, u, v, beta, d, l, tol, max_iter, verbose, uv_is_init, u0));
    return rcpp_result_gen;
END_RCPP
}
// predict_uv_R
Rcpp::List predict_uv_R(const arma::mat& A, const arma::colvec& u_train, const arma::colvec& v_train, arma::colvec u_test, arma::colvec v_test, double tol, int max_iter);
RcppExport SEXP _SuperCENT_predict_uv_R(SEXP ASEXP, SEXP u_trainSEXP, SEXP v_trainSEXP, SEXP u_testSEXP, SEXP v_testSEXP, SEXP tolSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type u_train(u_trainSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type v_train(v_trainSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type u_test(u_testSEXP);
    Rcpp::traits::input_parameter< arma::colvec >::type v_test(v_testSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(predict_uv_R(A, u_train, v_train, u_test, v_test, tol, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// two_stage_c
Rcpp::List two_stage_c(const arma::mat& A, const arma::mat& X, const arma::colvec& y, double tol, int max_iter);
RcppExport SEXP _SuperCENT_two_stage_c(SEXP ASEXP, SEXP XSEXP, SEXP ySEXP, SEXP tolSEXP, SEXP max_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::colvec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(two_stage_c(A, X, y, tol, max_iter));
    return rcpp_result_gen;
END_RCPP
}
// test_rirlba
Rcpp::List test_rirlba(const arma::mat& A);
RcppExport SEXP _SuperCENT_test_rirlba(SEXP ASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    rcpp_result_gen = Rcpp::wrap(test_rirlba(A));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_SuperCENT_rcpparma_hello_world", (DL_FUNC) &_SuperCENT_rcpparma_hello_world, 0},
    {"_SuperCENT_rcpparma_outerproduct", (DL_FUNC) &_SuperCENT_rcpparma_outerproduct, 1},
    {"_SuperCENT_rcpparma_innerproduct", (DL_FUNC) &_SuperCENT_rcpparma_innerproduct, 1},
    {"_SuperCENT_rcpparma_bothproducts", (DL_FUNC) &_SuperCENT_rcpparma_bothproducts, 1},
    {"_SuperCENT_cent", (DL_FUNC) &_SuperCENT_cent, 9},
    {"_SuperCENT_cv_oracle_cent", (DL_FUNC) &_SuperCENT_cv_oracle_cent, 19},
    {"_SuperCENT_cv_cent", (DL_FUNC) &_SuperCENT_cv_cent, 15},
    {"_SuperCENT_lr", (DL_FUNC) &_SuperCENT_lr, 8},
    {"_SuperCENT_cv_oracle_lr", (DL_FUNC) &_SuperCENT_cv_oracle_lr, 19},
    {"_SuperCENT_cv_lr", (DL_FUNC) &_SuperCENT_cv_lr, 15},
    {"_SuperCENT_cv_lr_2", (DL_FUNC) &_SuperCENT_cv_lr_2, 15},
    {"_SuperCENT_predict", (DL_FUNC) &_SuperCENT_predict, 7},
    {"_SuperCENT_predict_svd", (DL_FUNC) &_SuperCENT_predict_svd, 3},
    {"_SuperCENT_predict_oracle", (DL_FUNC) &_SuperCENT_predict_oracle, 5},
    {"_SuperCENT_lr_check", (DL_FUNC) &_SuperCENT_lr_check, 13},
    {"_SuperCENT_predict_uv_R", (DL_FUNC) &_SuperCENT_predict_uv_R, 7},
    {"_SuperCENT_two_stage_c", (DL_FUNC) &_SuperCENT_two_stage_c, 5},
    {"_SuperCENT_test_rirlba", (DL_FUNC) &_SuperCENT_test_rirlba, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_SuperCENT(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}