#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

#include <iostream>
using namespace Rcpp;
using namespace arma;

int irlba (
	arma::mat& U, 
	arma::vec& s, 
	arma::mat& V,  
	const arma::mat& A,
	int k);

int irlba_eigen (
	arma::mat& U, 
	arma::vec& s, 
	const arma::mat& A,
	int k);

int update_beta(const arma::mat& X,
				const arma::colvec& u, 
				const arma::colvec& v, 
				const arma::colvec& y, 
				arma::colvec& beta);

int update_beta(const arma::mat& X,
				const arma::colvec& u, 
				const arma::colvec& y, 
				arma::colvec& beta);

int update_beta_2(const arma::mat& X,
				const arma::colvec& u, 
				const arma::colvec& v, 
				const arma::colvec& y, 
				arma::colvec& beta);

double update_d(const arma::mat& A,
			const arma::colvec& u, 
			const arma::colvec& v);

double update_d_cent(const arma::mat& A,
			const arma::colvec& u, 
			const arma::colvec& v, 
			double l1, double l2);

int update_u(const arma::mat& A,
			const arma::mat& X,
			arma::colvec& u, 
			const arma::colvec& v, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l1, 
			double l2);

int update_v(const arma::mat& A,
			const arma::mat& X,
			const arma::colvec& u, 
			arma::colvec& v, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l1, 
			double l2);

int update_u_2(const arma::mat& A,
			const arma::mat& X,
			arma::colvec& u, 
			const arma::colvec& v, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l);

int update_v_2(const arma::mat& A,
			const arma::mat& X,
			const arma::colvec& u, 
			arma::colvec& v, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l);

int update_u_sym(const arma::mat& A,
			const arma::mat& X,
			arma::colvec& u, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l);

bool stop_condition(const arma::colvec& u, 
					const arma::colvec& u_old,
					const arma::colvec& v, 
					const arma::colvec& v_old,
					double tol);

int lr_(const arma::mat& A, 
				const arma::mat& X, 
				const arma::colvec& y,
				arma::colvec& u,
				arma::colvec& v,
				arma::colvec& beta,
				arma::colvec& u_distance,
				double& d,
				double l, 
				double tol,
				int max_iter,
				int verbose,
				bool uv_is_init);

int lr_sym(const arma::mat& A, 
		const arma::mat& X, 
		const arma::colvec& y,
		arma::colvec& u,
		arma::colvec& beta,
		arma::colvec& u_distance,
		double& d,
		double l, 
		double tol,
		int max_iter,
		int verbose,
		bool uv_is_init);

int cent_(const arma::mat& A, 
		const arma::mat& X, 
		const arma::colvec& y,
		arma::colvec& u,
		arma::colvec& v,
		arma::colvec& beta,
		double& d,
		double l1,
		double l2, 
		double tol,
		int max_iter,
		int verbose,
		bool uv_is_init);

arma::colvec predict(const arma::mat& A,
			const arma::mat& X_test,
			const arma::colvec& u_train, 
			const arma::colvec& v_train,
			const arma::colvec& beta,
			arma::colvec& u_test,
			arma::colvec& v_test,
			double tol,
			int max_iter);

arma::colvec predict(const arma::mat& A,
			const arma::mat& X_test,
			const arma::colvec& beta,
			arma::colvec& u_test,
			arma::colvec& v_test);

int predict_uv(const arma::mat& A,
			const arma::colvec& u_train, 
			const arma::colvec& v_train,
			arma::colvec& u_test,
			arma::colvec& v_test,
			double tol,
			int max_iter);

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List cent(const arma::mat& A, 
				const arma::mat& X, 
				const arma::colvec& y,
				double l1, 
				double l2,
				double tol,
				int max_iter,
				int verbose = 0,
				int scaled = 1) 
{
	int k = X.n_cols;
	
	// output
	arma::colvec u, v, beta(k + 2, fill::zeros), residuals(X.n_rows);
	double d = 0;
	int iter;
	
	iter = cent_(A, X, y, u, v, beta, d, l1, l2, tol, max_iter, verbose, false);

	residuals = y - join_rows(X, u, v) * beta;
	Rcpp::StringVector method = "cent";

	return Rcpp::List::create(_("u") = u,
							  _("v") = v,
							  _("beta") = beta,
							  _("d") = d,
							  _("iter") = iter,
							  _("residuals") = residuals,
							  _("method") = method);
}

// [[Rcpp::export]]
Rcpp::List cv_oracle_cent(const arma::mat& A, 
				const arma::mat& X, 
				const arma::colvec& y,
				double lmin, 
				double lmax,
				double gap,
				double tol,
				int max_iter,
				const arma::colvec& beta0,
				int verbose = 0,
				int scaled = 1,
				int scaledn = 0,
				Nullable<NumericVector> d0_ = R_NilValue,
				Nullable<NumericVector> u0_ = R_NilValue,
				Nullable<NumericVector> v0_ = R_NilValue,
				Nullable<NumericMatrix> X_test_ = R_NilValue,
				Nullable<NumericVector> y_test_ = R_NilValue,
				Nullable<NumericVector> u_test0_ = R_NilValue,
				Nullable<NumericVector> v_test0_ = R_NilValue) 
{
	int n = X.n_rows, k = X.n_cols, n_ll = std::floor((lmax-lmin)/gap) + 1;

	arma::colvec u0, v0, d0;
    if(u0_.isNotNull() & v0_.isNotNull()) {
    	u0 = as<arma::colvec>(u0_);
    	v0 = as<arma::colvec>(v0_);
    	d0 = as<arma::colvec>(d0_);
	}

	arma::colvec u_test0, v_test0, y_test;
    arma::mat X_test;
    if(X_test_.isNotNull()) {
    	X_test = as<arma::mat>(X_test_);
    	y_test = as<arma::colvec>(y_test_);	
    } 
	if(u_test0_.isNotNull() & u_test0_.isNotNull()) {
		u_test0 = as<arma::colvec>(u_test0_);
    	v_test0 = as<arma::colvec>(v_test0_);
	}
	
	// output
	arma::colvec u_min, v_min, beta_min(k + 2, fill::zeros);
	arma::mat beta_l2(n_ll, n_ll, fill::zeros);
	double d_min = 0, l1_min, l2_min;
	int iter_min;

	arma::colvec u_testing_min, v_testing_min, beta_testing_min(k + 2, fill::zeros), residuals_testing(X.n_rows);
	arma::mat testing_error(n_ll, n_ll, fill::zeros);
	double d_testing_min = 0, l1_testing_min, l2_testing_min, testing_err_min;
	int iter_testing_min;

	// aux
	arma::colvec u, v, beta(k + 2, fill::zeros), residuals(X.n_rows);
	double d = 0;

	arma::mat U, V;
	arma::vec s;

	arma::vec ll_sequence = arma::linspace<arma::vec>(lmin, lmax, n_ll);
	ll_sequence = arma::exp2(ll_sequence);
	
	// initiate u, v
	arma::svd(U, s, V, A);

	if(u0_.isNotNull() & v0_.isNotNull()) {
		u_min = u0; v_min = v0; d_min = d0(0);
	} else {
		u_min = U.col(0)*sqrt(n); v_min = V.col(0)*sqrt(n); d_min = s(0)/n; 
	}

	l1_min = -1; l2_min = -1; iter_min = 0;
	beta_min = arma::solve(join_rows(X, u_min, v_min), y);

	if(X_test_.isNotNull()) {
		u_testing_min = u_min; v_testing_min = v_min; d_testing_min = d_min;
		beta_testing_min = beta_min;
		l1_testing_min = -1; l2_testing_min = -1; iter_testing_min = 0;
		testing_err_min = sum(pow(y_test - join_rows(X_test, u_test0, v_test0) * beta_min, 2));
	}


	for(int ll1_iter = 0; ll1_iter < n_ll; ll1_iter++) {

		double ll1 = ll_sequence(ll1_iter);

		for(int ll2_iter = 0; ll2_iter < n_ll; ll2_iter++) {

			double ll2 = ll_sequence(ll2_iter);

			if((verbose == 1) & (ll1_iter % 10 == 0) & (ll2_iter == lmin)) std::cout << "ll: (" << ll1_iter << ", " << ll2_iter << ")" << std::endl;

			if(u0_.isNotNull() & v0_.isNotNull()) {
				u = u0; v = v0; d = d0(0);
			} else {
				u = U.col(0)*sqrt(n); v = V.col(0)*sqrt(n); d = s(0)/n; 
			}

			beta.zeros();

			int iter = cent_(A, X, y, u, v, beta, d, ll1, ll2, tol, max_iter, verbose, true);

			// oracle
			beta_l2(ll1_iter, ll2_iter) = norm(beta0 - beta);

			if(verbose) std::cout << "\t" << beta.t() << std::endl;
			if(norm(beta0 - beta) < norm(beta0 - beta_min)) {
				if(verbose == 3) std::cout << "\t Update: " << ll1_iter << ", " << ll2_iter;
				beta_min = beta; u_min = u; v_min = v; d_min = d; l1_min = ll1; l2_min = ll2; iter_min = iter;
			}

			if(X_test_.isNotNull()) {
				residuals_testing = y_test - join_rows(X_test, u_test0, v_test0) * beta;
				double testing_err = sum(square(residuals_testing));
				testing_error(ll1_iter, ll2_iter) = testing_err;
				if(testing_err < testing_err_min) {
					beta_testing_min = beta; 
					u_testing_min = u; v_testing_min = v; d_testing_min = d; 
					l1_testing_min = ll1; l2_testing_min = ll2; iter_testing_min = iter;
					testing_err_min = testing_err; 
				}
			}

		}
	}
	
	residuals = y - join_rows(X, u_min, v_min) * beta_min;
	residuals_testing = y - arma::join_rows(X, u_testing_min, v_testing_min) * beta_testing_min;	

	Rcpp::StringVector method = "cv_oracle_cent_min_beta_l2";
	Rcpp::List ret_beta = Rcpp::List::create(_("u") = u_min,
							  _("v") = v_min,
							  _("beta") = beta_min,
							  _("d") = d_min,
							  _("l1") = l1_min,
							  _("l2") = l2_min,
							  _("beta_l2") = beta_l2,
							  _("iter") = iter_min,
							  _("residuals") = residuals,
							  _("method") = method);

	Rcpp::List ret = Rcpp::List::create(_("min_beta_l2") = ret_beta);

	if(X_test_.isNotNull()) {
		Rcpp::StringVector method_testing = "cv_oracle_cent";
		Rcpp::List ret_testing = Rcpp::List::create(_("u") = u_testing_min,
							  _("v") = v_testing_min,
							  _("beta") = beta_testing_min,
							  _("d") = d_testing_min,
							  _("l1") = l1_testing_min,
							  _("l2") = l2_testing_min,
							  _("testing_error") = testing_error,
							  _("iter") = iter_testing_min,
							  _("residuals") = residuals_testing,
							  _("method") = method_testing);

		ret = Rcpp::List::create(_("min_beta_l2") = ret_beta,
								_("min_testing") = ret_testing);
	}

	return ret;

}

// [[Rcpp::export]]
Rcpp::List cv_cent(const arma::mat& A, 
				const arma::mat& X, 
				const arma::colvec& y,
				double lmin, 
				double lmax,
				double gap,
				double tol,
				int max_iter,
				int folds = 10,
				int verbose = 0,
				int scaled = 1,
				int scaledn = 0,
				Nullable<NumericVector> d0_ = R_NilValue,
				Nullable<NumericVector> u0_ = R_NilValue,
				Nullable<NumericVector> v0_ = R_NilValue) 
{
	int n = X.n_rows, k = X.n_cols, n_ll = std::floor((lmax-lmin)/gap) + 1;

	arma::colvec u0, v0, d0;
    if(u0_.isNotNull() & v0_.isNotNull()) {
    	u0 = as<arma::colvec>(u0_);
    	v0 = as<arma::colvec>(v0_);
    	d0 = as<arma::colvec>(d0_);
	}
	
	// cv output
	arma::colvec u_min, v_min, beta_min(k + 2, fill::zeros);
	arma::mat mse_cv(n_ll, n_ll, fill::zeros);
	double d_min = 0, l1_min, l2_min;
	int iter_min;
	
	// cv aux
	arma::mat A_cv, X_cv, U_cv, V_cv;
	arma::vec s_cv;
	arma::colvec u_cv, v_cv, beta_cv(k + 2, fill::zeros), residuals(X.n_rows);
	arma::colvec y_cv, u_test, v_test;
	// arma::colvec u_old, v_old, beta_old;
	// arma::uvec sequence = arma::linspace<arma::uvec>(0, folds-1, folds);
	// arma::uvec cv_index = Rcpp::RcppArmadillo::sample(sequence, X.n_rows, true);
	arma::uvec cv_index(n, fill::zeros);
	for(int fold = 1; fold < folds; fold++) {cv_index.subvec(fold*n/folds, n-1) += 1;}
	arma::vec ll_sequence = arma::linspace<arma::vec>(lmin, lmax, n_ll);
	ll_sequence = arma::exp2(ll_sequence);
	double d_cv;
	
	// cv
	for(int fold = 0; fold < folds; fold++) {

		if(verbose) std::cout << "fold " << fold << std::endl;

		A_cv = A(find(cv_index != fold), find(cv_index != fold));
		X_cv = X.rows(find(cv_index != fold)); 
		y_cv = y(find(cv_index != fold));

		arma::svd(U_cv, s_cv, V_cv, A_cv);

		for(int ll1_iter = 0; ll1_iter < n_ll; ll1_iter++) {

			double ll1 = ll_sequence(ll1_iter);

			for(int ll2_iter = 0; ll2_iter < n_ll; ll2_iter++) {

				double ll2 = ll_sequence(ll2_iter);

				if((verbose == 1) & (ll1_iter % 10 == 0) & (ll2_iter == lmin)) std::cout << "(" << ll1_iter << ", " << ll2_iter << ")" << std::endl;

				if(u0_.isNotNull() & v0_.isNotNull()) {
					u_cv = u0(find(cv_index != fold)); v_cv = v0(find(cv_index != fold)); d_cv = d0(0);
				} else {
					u_cv = U_cv.col(0)*sqrt(n); v_cv = V_cv.col(0)*sqrt(n); d_cv = s_cv(0)/n; 
				}
				beta_cv.zeros();

				cent_(A_cv, X_cv, y_cv, u_cv, v_cv, beta_cv, d_cv, ll1, ll2, tol, max_iter, verbose, true);

				// testing error
				double mse_tmp;
				if(u0_.isNotNull() & v0_.isNotNull()) {
					arma::colvec y_hat = join_rows(X.rows(find(cv_index == fold)), u0(find(cv_index == fold)), v0(find(cv_index == fold))) * beta_cv; 
					mse_tmp = norm(y(find(cv_index == fold)) - y_hat, "fro");
				} else {
					arma::colvec y_hat = predict(A, X.rows(find(cv_index == fold)), u_cv, v_cv, beta_cv, u_test, v_test, tol, max_iter);
					mse_tmp = norm(y(find(cv_index == fold)) - y_hat, "fro");
				}

				// mse_cv
				if(fold == 0) {
					mse_cv(ll1_iter, ll2_iter) = mse_tmp;
				} else {
					mse_cv(ll1_iter, ll2_iter) += (mse_tmp - mse_cv(ll1_iter, ll2_iter))/(fold + 1);
				}
			}
		}
	}

	l1_min = ll_sequence(mse_cv.index_min() % n_ll);
	l2_min = ll_sequence(mse_cv.index_min() / n_ll);

	// std::cout << mse_cv.index_min() % n_ll << ", " << mse_cv.index_min() / n_ll << std::endl;

	bool uv_is_init = false;
	if(u0_.isNotNull() & v0_.isNotNull()) {u_min = u0; v_min = v0; d_min = d0(0); uv_is_init = true; }
	iter_min = cent_(A, X, y, u_min, v_min, beta_min, d_min, l1_min, l2_min, tol, max_iter, verbose, uv_is_init);
	
	residuals = y - join_rows(X, u_min, v_min) * beta_min;

	Rcpp::StringVector method = "cv_oracle_cent";

	return Rcpp::List::create(_("u") = u_min,
							  _("v") = v_min,
							  _("beta") = beta_min,
							  _("d") = d_min,
							  _("l1") = l1_min,
							  _("l2") = l2_min,
							  _("mse_cv") = mse_cv,
							  _("iter") = iter_min,
							  _("residuals") = residuals,
							  _("cv_index") = cv_index,
							  _("method") = method);
}


// [[Rcpp::export]]
Rcpp::List lr(const arma::mat& A, 
				const arma::mat& X, 
				const arma::colvec& y,
				double l, 
				double tol,
				int max_iter,
				int verbose = 0,
				int scaled = 1) 
{
	int n_obs = X.n_rows, n = A.n_rows, k = X.n_cols;

	// output
	Rcpp::List ret_list;
	arma::colvec u, v, beta(k + 2, fill::zeros), residuals(n), u_distance(max_iter);
	double d;
	int iter;
	Rcpp::StringVector method = "lr";

	if(A.is_symmetric()) {

		std::cout << "Symmetric matrix" << std::endl;

		iter = lr_sym(A, X, y, u, beta, u_distance, d, l, tol, max_iter, verbose, false);

		residuals = y - join_rows(X, u.head(n_obs)) * beta;

		ret_list = Rcpp::List::create(_("u") = u,
							  _("beta") = beta,
							  _("d") = d,
							  _("l") = l,
							  _("iter") = iter,
							  _("residuals") = residuals,
							  _("method") = method,
							  _("max_iter") = max_iter,
							  _("u_distance") = u_distance.head(iter));

	} else {

		iter = lr_(A, X, y, u, v, beta, u_distance, d, l, tol, max_iter, verbose, false);

		residuals = y - join_rows(X, u.head(n_obs), v.head(n_obs)) * beta;

		ret_list = Rcpp::List::create(_("u") = u,
							  _("v") = v,
							  _("beta") = beta,
							  _("d") = d,
							  _("l") = l,
							  _("iter") = iter,
							  _("residuals") = residuals,
							  _("method") = method,
							  _("max_iter") = max_iter,
							  _("u_distance") = u_distance.head(iter));
	}
	
	return ret_list;
}

// [[Rcpp::export]]
Rcpp::List cv_oracle_lr(const arma::mat& A, 
				const arma::mat& X, 
				const arma::colvec& y,
				double lmin, 
				double lmax,
				double gap,
				double tol,
				int max_iter,
				const arma::colvec& beta0,
				int verbose = 0,
				int scaled = 1,
				int scaledn = 0,
				Nullable<NumericVector> d0_ = R_NilValue,
				Nullable<NumericVector> u0_ = R_NilValue,
				Nullable<NumericVector> v0_ = R_NilValue,
				Nullable<NumericMatrix> X_test_ = R_NilValue,
				Nullable<NumericVector> y_test_ = R_NilValue,
				Nullable<NumericVector> u_test0_ = R_NilValue,
				Nullable<NumericVector> v_test0_ = R_NilValue) 
{
	int n = X.n_rows, k = X.n_cols, n_ll = std::floor((lmax-lmin)/gap) + 1;

    arma::colvec u0, v0, d0;
    if(u0_.isNotNull() & v0_.isNotNull()) {
    	u0 = as<arma::colvec>(u0_);
    	v0 = as<arma::colvec>(v0_);
    	d0 = as<arma::colvec>(d0_);
	}

	arma::colvec u_test0, v_test0, y_test;
    arma::mat X_test;
    if(X_test_.isNotNull()) {
    	X_test = as<arma::mat>(X_test_);
    	y_test = as<arma::colvec>(y_test_);	
    } 
	if(u_test0_.isNotNull() & u_test0_.isNotNull()) {
		u_test0 = as<arma::colvec>(u_test0_);
    	v_test0 = as<arma::colvec>(v_test0_);
	}

	// output
	arma::colvec u_min, v_min, beta_min(k + 2, fill::zeros), u_distance_min(max_iter);
	arma::colvec beta_l2(n_ll, fill::zeros);
	double d_min = 0, l_min;
	int iter_min;

	arma::colvec u_testing_min, v_testing_min, beta_testing_min(k + 2, fill::zeros), residuals_testing(X.n_rows), u_distance_testing_min(max_iter);
	arma::colvec testing_error(n_ll, fill::zeros);
	double d_testing_min = 0, l_testing_min, testing_err_min;
	int iter_testing_min;

	// aux
	arma::colvec u, v, beta(k + 2, fill::zeros), residuals(X.n_rows), u_distance(max_iter);
	double d = 0;

	arma::mat U, V;
	arma::vec s;

	arma::vec ll_sequence = arma::linspace<arma::vec>(lmin, lmax, n_ll);
	ll_sequence = arma::exp2(ll_sequence);
	// ll_sequence *= n;
	
	// initiate u, v
	arma::svd(U, s, V, A);

	if(u0_.isNotNull() & v0_.isNotNull()) {
		u_min = u0; v_min = v0; d_min = d0(0);
	} else {
		u_min = U.col(0)*sqrt(n); v_min = V.col(0)*sqrt(n); d_min = s(0)/n; 
	}

	l_min = -1; iter_min = 0;
	beta_min = arma::solve(join_rows(X, u_min, v_min), y);
	std::cout << "!!! beta min init " << norm(beta0 - beta_min) << std::endl;
	
	if(X_test_.isNotNull()) {
		u_testing_min = u_min; v_testing_min = v_min; d_testing_min = d_min;
		beta_testing_min = beta_min;
		l_testing_min = -1; iter_testing_min = 0;
		testing_err_min = sum(pow(y_test - join_rows(X_test, u_test0, v_test0) * beta_min, 2));
		std::cout << "!!! testing err min init " << testing_err_min << std::endl;
	}

	
	for(int ll_iter = 0; ll_iter < n_ll; ll_iter++) {

		double ll = ll_sequence(ll_iter);
		
		if(verbose) std::cout << "ll: " << ll << std::endl;

		int iter;

		if(u0_.isNotNull() & v0_.isNotNull()) {
			u = u0; v = v0; d = d0(0);
		} else {
			u = U.col(0)*sqrt(n); v = V.col(0)*sqrt(n); d = s(0)/n;
		}

		beta.zeros();

		iter = lr_(A, X, y, u, v, beta, u_distance, d, ll, tol, max_iter, verbose, true);

		// oracle
		beta_l2(ll_iter) = norm(beta0 - beta);

		if(verbose) std::cout << "\t" << beta.t() << std::endl;
		if(norm(beta0 - beta) < norm(beta0 - beta_min)) {
			if(verbose == 3) {
				std::cout << "\t Update" << iter << beta_min.t();
				std::cout << "\t" << iter << beta.t();
			}
			beta_min = beta; u_min = u; v_min = v; d_min = d; l_min = ll; iter_min = iter;
			u_distance_min = u_distance;
		}

		if(X_test_.isNotNull()) {
			residuals_testing = y_test - arma::join_rows(X_test, u_test0, v_test0) * beta;
			double testing_err = sum(square(residuals_testing));
			testing_error(ll_iter) = testing_err;
			if(testing_err < testing_err_min) {
				beta_testing_min = beta; 
				u_testing_min = u; v_testing_min = v; d_testing_min = d; 
				l_testing_min = ll; iter_testing_min = iter;
				testing_err_min = testing_err; 
				u_distance_testing_min = u_distance;
			}
		}

	}
	
	residuals = y - join_rows(X, u_min, v_min) * beta_min;
	residuals_testing = y - arma::join_rows(X, u_testing_min, v_testing_min) * beta_testing_min;	

	Rcpp::StringVector method = "cv_oracle_lr_min_beta_l2";
	Rcpp::List ret_beta = Rcpp::List::create(_("u") = u_min,
							  _("v") = v_min,
							  _("beta") = beta_min,
							  _("d") = d_min,
							  _("l") = l_min,
							  _("beta_l2") = beta_l2,
							  _("iter") = iter_min,
							  _("residuals") = residuals,
							  _("method") = method,
							  _("max_iter") = max_iter,
							  _("u_distance") = u_distance_min.head(iter_min));

	Rcpp::List ret = Rcpp::List::create(_("min_beta_l2") = ret_beta);

	if(X_test_.isNotNull()) {
		Rcpp::StringVector method_testing = "cv_oracle_lr";
		Rcpp::List ret_testing = Rcpp::List::create(_("u") = u_testing_min,
							  _("v") = v_testing_min,
							  _("beta") = beta_testing_min,
							  _("d") = d_testing_min,
							  _("l") = l_testing_min,
							  _("testing_error") = testing_error,
							  _("iter") = iter_testing_min,
							  _("residuals") = residuals_testing,
							  _("method") = method_testing,
							  _("max_iter") = max_iter,
							  _("u_distance") = u_distance_testing_min.head(iter_testing_min));

		ret = Rcpp::List::create(_("min_beta_l2") = ret_beta,
								_("min_testing") = ret_testing);
	}

	return ret;
}

// [[Rcpp::export]]
Rcpp::List cv_lr(const arma::mat& A, 
				const arma::mat& X, 
				const arma::colvec& y,
				double lmin, 
				double lmax,
				double gap,
				double tol,
				int max_iter,
				int folds = 10,
				int verbose = 0,
				int scaled = 1,
				int scaledn = 0,
				Nullable<NumericVector> d0_ = R_NilValue,
				Nullable<NumericVector> u0_ = R_NilValue,
				Nullable<NumericVector> v0_ = R_NilValue) 
{
	
	double llmin = floor(log2(lmin)), llmax = floor(log2(lmax));
	int n = X.n_rows, k = X.n_cols, n_ll = std::floor((llmax - llmin)/gap) + 1;

	arma::colvec u0, v0, d0;
    if(u0_.isNotNull() & v0_.isNotNull()) {
    	u0 = as<arma::colvec>(u0_);
    	v0 = as<arma::colvec>(v0_);
    	d0 = as<arma::colvec>(d0_);
	}
	
	// cv output
	arma::colvec u_min, v_min, beta_min(k + 2, fill::zeros), u_distance_min(max_iter);
	arma::colvec mse_cv(n_ll, fill::zeros);
	double d_min = 0, l_min;
	int iter_min;
	
	arma::mat beta_cvs(folds, k+2, fill::zeros);

	// cv aux
	arma::mat A_cv, X_cv, U_cv, V_cv, A_;
	A_.set_size(size(A));
	arma::vec s_cv;
	arma::colvec u_cv, v_cv, beta_cv(k + 2, fill::zeros), residuals(X.n_rows);
	arma::colvec y_cv, u_test, v_test;
	// arma::colvec u_old, v_old, beta_old;
	// arma::uvec sequence = arma::linspace<arma::uvec>(0, folds-1, folds);
	// arma::uvec cv_index = Rcpp::RcppArmadillo::sample(sequence, X.n_rows, true);
	arma::uvec cv_index(n, fill::zeros);
	for(int fold = 1; fold < folds; fold++) {cv_index.subvec(fold*n/folds, n-1) += 1;}
	arma::vec ll_sequence = arma::linspace<arma::vec>(llmin, llmax, n_ll);
	ll_sequence = arma::exp2(ll_sequence);
	// ll_sequence *= n;
	double d_cv;

	// cv
	for(int fold = 0; fold < folds; fold++) {

		if(verbose) std::cout << "fold " << fold << std::endl;

		// training
		// std::cout << sum(cv_index != fold) << std::endl;
		A_cv = A(find(cv_index != fold), find(cv_index != fold));
		X_cv = X.rows(find(cv_index != fold)); 
		y_cv = y(find(cv_index != fold));

		int cv_size = sum(cv_index != fold), fold_size = sum(cv_index == fold);
		A_.submat(0, 0, size(cv_size, cv_size)) = A_cv;
		A_.submat(cv_size, cv_size, size(fold_size, fold_size)) = A(find(cv_index == fold), find(cv_index == fold));
		A_.submat(cv_size, 0, size(fold_size, cv_size)) = A(find(cv_index == fold), find(cv_index != fold));
		A_.submat(0, cv_size, size(cv_size, fold_size)) = A(find(cv_index != fold), find(cv_index == fold));

		irlba(U_cv, s_cv, V_cv, A_cv, 1);
		
		for(int ll_iter = 0 ; ll_iter < n_ll; ll_iter++) {

			double ll = ll_sequence(ll_iter);

			if(verbose) std::cout << "ll: " << ll << std::endl;

			if(u0_.isNotNull() & v0_.isNotNull()) {
				u_cv = u0(find(cv_index != fold)); v_cv = v0(find(cv_index != fold)); d_cv = d0(0);
			} else {
				u_cv = U_cv.col(0)*sqrt(n); v_cv = V_cv.col(0)*sqrt(n); d_cv = s_cv(0)/n; 
			}
			beta_cv.zeros();

			lr_(A_cv, X_cv, y_cv, u_cv, v_cv, beta_cv, u_distance_min, d_cv, ll, tol, max_iter, verbose, true);

			// testing error
			double mse_tmp;
			if(u0_.isNotNull() & v0_.isNotNull()) {
				arma::colvec y_hat = join_rows(X.rows(find(cv_index == fold)), u0(find(cv_index == fold)), v0(find(cv_index == fold))) * beta_cv; 
				mse_tmp = norm(y(find(cv_index == fold)) - y_hat, "fro");
			} else {
				// arma::colvec y_hat = predict(A, X.rows(find(cv_index == fold)), u_cv, v_cv, beta_cv, u_test, v_test, tol, max_iter);
				arma::colvec y_hat = predict(A_, X.rows(find(cv_index == fold)), beta_cv, u_test, v_test);
				mse_tmp = norm(y(find(cv_index == fold)) - y_hat, "fro");
			}

			beta_cvs.row(fold) = beta_cv.t();

			// mse_cv
			if(fold == 0) {
				mse_cv(ll_iter) = mse_tmp;
			} else {
				mse_cv(ll_iter) += (mse_tmp - mse_cv(ll_iter))/(fold + 1);
			}
			
			// if(ll_iter % 10 == 0) std::cout << ll_iter << std::endl;
		}
	}

	l_min = ll_sequence(mse_cv.index_min());

	bool uv_is_init = false;
	if(u0_.isNotNull() & v0_.isNotNull()) {u_min = u0; v_min = v0; d_min = d0(0); uv_is_init = true; }
	iter_min = lr_(A, X, y, u_min, v_min, beta_min, u_distance_min, d_min, l_min, tol, max_iter, verbose, uv_is_init);
	
	residuals = y - join_rows(X, u_min, v_min) * beta_min;
	
	Rcpp::StringVector method = "cv_lr";

	return Rcpp::List::create(_("u") = u_min,
							  _("v") = v_min,
							  _("beta") = beta_min,
							  _("d") = d_min,
							  _("l") = l_min,
							  _("mse_cv") = mse_cv,
							  _("l_sequence") = ll_sequence,
							  _("iter") = iter_min,
							  _("residuals") = residuals,
							  _("cv_index") = cv_index,
							  _("beta_cvs") = beta_cvs,
							  _("method") = method,
							  _("max_iter") = max_iter,
							  _("u_distance") = u_distance_min.head(iter_min));
}

// [[Rcpp::export]]
Rcpp::List cv_lr_2(const arma::mat& A, 
				const arma::mat& X, 
				const arma::colvec& y,
				double lmin, 
				double lmax,
				double gap,
				double tol,
				int max_iter,
				int folds = 10,
				int verbose = 0,
				int scaled = 1,
				int scaledn = 0,
				Nullable<NumericVector> d0_ = R_NilValue,
				Nullable<NumericVector> u0_ = R_NilValue,
				Nullable<NumericVector> v0_ = R_NilValue) 
{
	
	double llmin = floor(log2(lmin)), llmax = floor(log2(lmax));
	int n = A.n_rows, n_obs = X.n_rows, k = X.n_cols, n_c = n - n_obs;
	int n_ll = std::floor((llmax - llmin)/gap) + 1;

	arma::colvec u0, v0, d0;
    if(u0_.isNotNull() & v0_.isNotNull()) {
    	u0 = as<arma::colvec>(u0_);
    	v0 = as<arma::colvec>(v0_);
    	d0 = as<arma::colvec>(d0_);
	}
	
	// cv output
	arma::colvec u_min, v_min, beta_min(k + 2, fill::zeros), u_distance_min(max_iter);
	arma::colvec mse_cv(n_ll, fill::zeros);
	double d_min = 0, l_min;
	int iter_min;
	
	arma::mat beta_cvs(folds, k+2, fill::zeros);

	// cv aux
	arma::mat A_cv, X_cv, U_cv, V_cv;
	A_cv.set_size(size(A));
	arma::vec s_cv;
	arma::colvec u_cv, v_cv, beta_cv(k + 2, fill::zeros), residuals(X.n_rows);
	arma::colvec y_cv, u_test, v_test;
	// arma::colvec u_old, v_old, beta_old;
	// arma::uvec sequence = arma::linspace<arma::uvec>(0, folds-1, folds);
	// arma::uvec cv_index = Rcpp::RcppArmadillo::sample(sequence, X.n_rows, true);
	arma::uvec cv_index(n_obs, fill::zeros);
	for(int fold = 1; fold < folds; fold++) {cv_index.subvec(fold*n_obs/folds, n_obs-1) += 1;}
	arma::vec ll_sequence = arma::linspace<arma::vec>(llmin, llmax, n_ll);
	ll_sequence = arma::exp2(ll_sequence);
	double d_cv;

	// cv
	for(int fold = 0; fold < folds; fold++) {

		if(verbose) std::cout << "fold " << fold << std::endl;

		// training
		int cv_size = sum(cv_index != fold), fold_size = sum(cv_index == fold);
		// A: diag 
		A_cv.submat(0, 0, size(cv_size, cv_size)) = A(find(cv_index != fold), find(cv_index != fold));
		A_cv.submat(cv_size, cv_size, size(fold_size, fold_size)) = A(find(cv_index == fold), find(cv_index == fold));
		A_cv.submat(cv_size+fold_size, cv_size+fold_size, size(n_c, n_c)) = A(n_obs, n_obs, size(n_c, n_c));
		// A21
		A_cv.submat(cv_size, 0, size(fold_size, cv_size)) = A(find(cv_index == fold), find(cv_index != fold) );
		// A31
		arma::mat tmp = A.rows(n_obs, n-1);
		tmp = tmp.cols(find(cv_index != fold));
		A_cv.submat(cv_size+fold_size, 0, size(n_c, cv_size)) = tmp;
		// A12
		A_cv.submat(0, cv_size, size(cv_size, fold_size)) = A(find(cv_index != fold), find(cv_index == fold));
		// A13
		tmp.reset();
		tmp = A.cols(n_obs, n-1);
		tmp = tmp.rows(find(cv_index != fold));
		A_cv.submat(0, cv_size+fold_size, size(cv_size, n_c)) = tmp;
		// A23
		tmp.reset();
		tmp = A.cols(n_obs, n-1);
		tmp = tmp.rows(find(cv_index == fold));
		A_cv.submat(cv_size, cv_size+fold_size, size(fold_size, n_c)) = tmp;
		// A32
		tmp.reset();
		tmp = A.rows(n_obs, n-1);
		tmp = tmp.cols(find(cv_index == fold));
		A_cv.submat(cv_size+fold_size, cv_size, size(n_c, fold_size)) = tmp;
		

		// std::cout << A_cv << std::endl;
		
		X_cv = X.rows(find(cv_index != fold)); 
		y_cv = y(find(cv_index != fold));

		irlba(U_cv, s_cv, V_cv, A_cv, 1);
		
		for(int ll_iter = 0 ; ll_iter < n_ll; ll_iter++) {

			double ll = ll_sequence(ll_iter);

			if(verbose) std::cout << "ll: " << ll << std::endl;

			if(u0_.isNotNull() & v0_.isNotNull()) {
				u_cv = u0(find(cv_index != fold)); v_cv = v0(find(cv_index != fold)); d_cv = d0(0);
			} else {
				u_cv = U_cv.col(0)*sqrt(n); v_cv = V_cv.col(0)*sqrt(n); d_cv = s_cv(0)/n; 
			}
			beta_cv.zeros();

			lr_(A_cv, X_cv, y_cv, u_cv, v_cv, beta_cv, u_distance_min, d_cv, ll, tol, max_iter, verbose, true);

			// testing error
			double mse_tmp;
			if(u0_.isNotNull() & v0_.isNotNull()) {
				arma::colvec y_hat = join_rows(X.rows(find(cv_index == fold)), u0(find(cv_index == fold)), v0(find(cv_index == fold))) * beta_cv; 
				mse_tmp = norm(y(find(cv_index == fold)) - y_hat, "fro");
			} else {
				// arma::colvec y_hat = predict(A, X.rows(find(cv_index == fold)), u_cv, v_cv, beta_cv, u_test, v_test, tol, max_iter);
				// arma::colvec y_hat = predict(A, X.rows(find(cv_index == fold)), beta_cv, u_test, v_test);
				arma::colvec y_hat = join_rows(X.rows(find(cv_index == fold)), u_cv.tail(fold_size), v_cv.tail(fold_size)) * beta_cv;
				mse_tmp = norm(y(find(cv_index == fold)) - y_hat, "fro");
			}

			beta_cvs.row(fold) = beta_cv.t();

			// mse_cv
			if(fold == 0) {
				mse_cv(ll_iter) = mse_tmp;
			} else {
				mse_cv(ll_iter) += (mse_tmp - mse_cv(ll_iter))/(fold + 1);
			}
			
			// if(ll_iter % 10 == 0) std::cout << ll_iter << std::endl;
		}
	}

	l_min = ll_sequence(mse_cv.index_min());

	bool uv_is_init = false;
	if(u0_.isNotNull() & v0_.isNotNull()) {u_min = u0; v_min = v0; d_min = d0(0); uv_is_init = true; }
	iter_min = lr_(A, X, y, u_min, v_min, beta_min, u_distance_min, d_min, l_min, tol, max_iter, verbose, uv_is_init);
	
	residuals = y - join_rows(X, u_min.head(n_obs), v_min.head(n_obs)) * beta_min;
	
	Rcpp::StringVector method = "cv_lr";

	return Rcpp::List::create(_("u") = u_min,
							  _("v") = v_min,
							  _("beta") = beta_min,
							  _("d") = d_min,
							  _("l") = l_min,
							  _("mse_cv") = mse_cv,
							  _("l_sequence") = ll_sequence,
							  _("iter") = iter_min,
							  _("residuals") = residuals,
							  _("cv_index") = cv_index,
							  _("beta_cvs") = beta_cvs,
							  _("method") = method,
							  _("max_iter") = max_iter,
							  _("u_distance") = u_distance_min.head(iter_min));
}

// [[Rcpp::export]]
Rcpp::List predict(const arma::mat& A,
			const arma::colvec& u_train, 
			const arma::colvec& v_train,
			const arma::colvec& beta,
			const arma::mat& X_test,
			double tol,
			int max_iter)
{
	arma::colvec y_hat, u_test, v_test;

	y_hat = predict(A, X_test, u_train, v_train, beta, u_test, v_test, tol, max_iter); 

	return Rcpp::List::create(_("u") = u_test,
							  _("v") = v_test,
							  _("y") = y_hat);
}


// [[Rcpp::export]]
Rcpp::List predict_svd(const arma::mat& A,
			const arma::colvec& beta,
			const arma::mat& X_test)
{
	arma::colvec y_hat, u_test, v_test;

	y_hat = predict(A, X_test, beta, u_test, v_test); 

	return Rcpp::List::create(_("u") = u_test,
							  _("v") = v_test,
							  _("y") = y_hat);
}

// [[Rcpp::export]]
Rcpp::List predict_oracle(const arma::mat& A,
			const arma::colvec& beta,
			const arma::mat& X_test,
			const arma::colvec& u_test, 
			const arma::colvec& v_test)
{
	arma::colvec y_hat;
	// arma::colvec beta = ret["beta"];

	y_hat = join_rows(X_test, u_test, v_test) * beta;

	return Rcpp::List::create(_("u") = u_test,
							  _("v") = v_test,
							  _("y") = y_hat);
}


int cent_(const arma::mat& A, 
		const arma::mat& X, 
		const arma::colvec& y,
		arma::colvec& u,
		arma::colvec& v,
		arma::colvec& beta,
		double& d,
		double l1,
		double l2, 
		double tol,
		int max_iter,
		int verbose,
		bool uv_is_init) 
{
	int n = X.n_rows;
	
	// aux
	bool cond = 1; 
	arma::colvec u_old, v_old, beta_old;
	double d_old = 0;
	int iter = 0;
	
	// initiate u, v
	if(!uv_is_init) {
		arma::mat U, V;
		arma::vec s;
		arma::svd(U, s, V, A);
		u = U.col(0)*sqrt(n); v = V.col(0)*sqrt(n); d = s(0)/n;
	}

	while(cond & (iter < max_iter)) {
		u_old = u; v_old = v; beta_old = beta; d_old = d;
		update_beta(X, u, v, y, beta);
		d = update_d(A, u, v);
		d /= n;
		update_u(A, X, u, v, y, beta, d, l1, l2);
		update_v(A, X, u, v, y, beta, d, l1, l2);

		if(verbose) {
			std::cout << "\t" << iter << beta.t() << std::endl;
			std::cout << "\t" << "u: " << norm(u_old * u_old.t() - u * u.t(), 2) << "; v: " << norm(v_old * v_old.t() - v * v.t(), 2) << "; beta: " << norm(beta - beta_old) << "; d: " << d << std::endl;
		}
		
		cond = stop_condition(u, u_old, v, v_old, tol);

		iter++;
	}

	return iter;
}


int lr_(const arma::mat& A, 
		const arma::mat& X, 
		const arma::colvec& y,
		arma::colvec& u,
		arma::colvec& v,
		arma::colvec& beta,
		arma::colvec& u_distance,
		double& d,
		double l, 
		double tol,
		int max_iter,
		int verbose,
		bool uv_is_init) 
{
	int n_obs = X.n_rows, n = A.n_rows;

	// aux
	bool cond = 1; 
	arma::colvec u_old, v_old, beta_old;
	double d_old = 0;
	int iter = 0;
	
	// initiate u, v
	if(!uv_is_init) {
		arma::mat U, V;
		arma::vec s;
		irlba(U, s, V, A, 1);
		u = U.col(0)*sqrt(n); v = V.col(0)*sqrt(n); d = s(0)/n;
	}

	while(cond & (iter < max_iter)) {
		u_old = u; v_old = v; beta_old = beta; d_old = d;
		update_beta(X, u, v, y, beta);
		d = update_d(A, u, v);
		d /= pow(n, 2);
		// if(d < 0) {u = -u; d = -d;}
		update_u_2(A, X, u, v, y, beta, d, l);
		update_v_2(A, X, u, v, y, beta, d, l);

		if(verbose) {
			std::cout << "\t\t" << iter << ": " << beta.t() << std::endl;
			if(verbose == 3) {
				std::cout << "\t\t" << "u: " << norm(u_old * u_old.t() - u * u.t(), 2)/n << "; v: " << norm(v_old * v_old.t() - v * v.t(), 2)/n << "; beta: " << norm(beta - beta_old) << "; d: " << sqrt(pow(d - d_old, 2)) << std::endl;
			}
		}

		u_distance(iter) = norm(u_old * u_old.t() - u * u.t(), 2)/n;
		
		cond = stop_condition(u, u_old, v, v_old, tol);

		iter++;
	}

	return iter;
}

int lr_sym(const arma::mat& A, 
		const arma::mat& X, 
		const arma::colvec& y,
		arma::colvec& u,
		arma::colvec& beta,
		arma::colvec& u_distance,
		double& d,
		double l, 
		double tol,
		int max_iter,
		int verbose,
		bool uv_is_init) 
{
	int n_obs = X.n_rows, n = A.n_rows;

	// aux
	bool cond = 1; 
	arma::colvec u_old, beta_old;
	double d_old = 0;
	int iter = 0;
	
	// initiate u, v
	if(!uv_is_init) {
		arma::mat U;
		arma::vec s;
		irlba_eigen(U, s, A, 1);
		u = U.col(0)*sqrt(n); d = s(0)/n;
	}

	while(cond & (iter < max_iter)) {
		u_old = u; beta_old = beta; d_old = d;
		update_beta(X, u, y, beta);
		d = as_scalar(u.t() * A * u);
		d /= pow(n, 2);
		// if(d < 0) {u = -u; d = -d;}
		update_u_sym(A, X, u, y, beta, d, l);
	
		if(verbose) {
			std::cout << "\t\t" << iter << ": " << beta.t() << std::endl;
			if(verbose == 3) {
				std::cout << "\t\t" << "u: " << norm(u_old * u_old.t() - u * u.t(), 2)/n << "; beta: " << norm(beta - beta_old) << "; d: " << sqrt(pow(d - d_old, 2)) << std::endl;
			}
		}
	
		u_distance(iter) = norm(u_old * u_old.t() - u * u.t(), 2)/n;
		
		cond = stop_condition(u, u_old, u, u_old, tol);
	
		iter++;
	}

	return iter;
}

// [[Rcpp::export]]
Rcpp::List lr_check(const arma::mat& A, 
		const arma::mat& X, 
		const arma::colvec& y,
		arma::colvec& u,
		arma::colvec& v,
		arma::colvec& beta,
		double d,
		double l, 
		double tol,
		int max_iter,
		int verbose,
		bool uv_is_init,
		arma::colvec& u0) 
{
	int n_obs = X.n_rows, n = A.n_rows;

	// aux
	bool cond = 1; 
	arma::colvec u_old, v_old, beta_old, u_distance(max_iter), u_hist(max_iter);
	double d_old = 0;
	int iter = 0;
	
	// initiate u, v
	if(!uv_is_init) {
		arma::mat U, V;
		arma::vec s;
		irlba(U, s, V, A, 1);
		u = U.col(0)*sqrt(n); v = V.col(0)*sqrt(n); d = s(0)/n;
	}

	while(cond & (iter < max_iter)) {
		u_old = u; v_old = v; beta_old = beta; d_old = d;
		update_beta(X, u, v, y, beta);
		d = update_d(A, u, v);
		d /= pow(n, 2);
		if(d < 0) {u = -u; d = -d;}
		update_u_2(A, X, u, v, y, beta, d, l);
		update_v_2(A, X, u, v, y, beta, d, l);

		if(verbose) {
			std::cout << "\t\t" << iter << ": " << beta.t() << std::endl;
			std::cout << "\t\t" << "d: " << d << std::endl;
			if(verbose == 3) {
				std::cout << "\t\t" << "u: " << norm(u_old * u_old.t() - u * u.t(), 2)/n << "; v: " << norm(v_old * v_old.t() - v * v.t(), 2)/n << "; beta: " << norm(beta - beta_old) << "; d: " << sqrt(pow(d - d_old, 2)) << std::endl;
			}
		}

		u_distance(iter) = norm(u_old * u_old.t() - u * u.t(), 2)/n;
		u_hist(iter) = norm(u0 * u0.t() - u * u.t(), 2)/n;

		std::cout << iter << ": u, u0 distance:" << u_hist(iter) << std::endl;
		
		cond = stop_condition(u, u_old, v, v_old, tol);

		iter++;
	}

	return Rcpp::List::create(_("u") = u,
							  _("v") = v,
							  _("beta") = beta,
							  _("d") = d,
							  _("l") = l,
							  _("iter") = iter,
							  _("max_iter") = max_iter,
							  _("u_distance") = u_distance.head(iter),
							  _("u0_distance") = u_hist.head(iter));
}



int update_beta(const arma::mat& X,
				const arma::colvec& u, 
				const arma::colvec& v, 
				const arma::colvec& y, 
				arma::colvec& beta) 
{
	int n_obs = X.n_rows;

	arma::mat WW = arma::join_rows(X, u.head(n_obs));
	arma::mat W = arma::join_rows(WW, v.head(n_obs));
	beta = arma::solve(W, y); 

	return 1;
}


int update_beta(const arma::mat& X,
				const arma::colvec& u, 
				const arma::colvec& y, 
				arma::colvec& beta) 
{
	int n_obs = X.n_rows;

	arma::mat W = arma::join_rows(X, u.head(n_obs));
	beta = arma::solve(W, y); 

	return 1;
}


int update_beta_2(const arma::mat& X,
				const arma::colvec& u, 
				const arma::colvec& v, 
				const arma::colvec& y, 
				arma::colvec& beta) 
{
	int n_obs = X.n_rows, k = X.n_cols;

	arma::mat WW = arma::join_rows(X, u.head(n_obs));
	arma::mat W = arma::join_rows(WW, v.head(n_obs));
	beta = arma::solve(W, y); 

	arma::colvec us = u.head(n_obs);
	arma::colvec vs = v.head(n_obs);

	double us_norm = norm(us, 2);
	us_norm *= us_norm;
	double vs_norm = norm(vs, 2);
	vs_norm *= vs_norm;

	// std::cout << "us norm: " << us_norm << std::endl;
	// std::cout << "vs norm: " << vs_norm << std::endl;

	beta(k) = as_scalar(us.t()*(y - X*beta.head(k) - vs*beta(k+1)))/us_norm;

	// std::cout << "betau: " << beta(k) << std::endl;

	beta(k+1) = as_scalar(vs.t()*(y - X*beta.head(k) - us*beta(k)))/vs_norm;

	// std::cout << "betav: " << beta(k+1) << std::endl;

	beta.head(k) = (X.t() * X).i() * X.t() * (y - us*beta(k) - vs*beta(k+1));

	// std::cout << "beta: " << beta.t() << std::endl;

	return 1;
}

double update_d(const arma::mat& A,
			const arma::colvec& u, 
			const arma::colvec& v) 
{
	double d = as_scalar(u.t() * A * v);
	return d;
}

int update_u(const arma::mat& A,
			const arma::mat& X,
			arma::colvec& u, 
			const arma::colvec& v, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l1, 
			double l2) 
{
	int n = X.n_rows, k = X.n_cols;
	arma::mat lmat = l2*A*A.t();
	lmat.diag() += pow(beta(k),2) + l1*pow(d,2);
	arma::mat rmat = beta(k)*(y - X*beta.head(k) - beta(k+1)*v) + (l1+l2)*d*A*v;
	u = solve(lmat, rmat);

	// normalize u
	u = arma::normalise(u) * sqrt(n);

	return 1;
}

int update_v(const arma::mat& A,
			const arma::mat& X,
			const arma::colvec& u, 
			arma::colvec& v, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l1, 
			double l2) 
{
	int n = X.n_rows, k = X.n_cols;
	arma::mat lmat = l1*A.t()*A;
	lmat.diag() += pow(beta(k+1),2) + l2*pow(d,2);
	arma::mat rmat = beta(k+1)*(y - X*beta.head(k) - beta(k)*u) + (l1+l2)*d*A.t()*u;
	v = solve(lmat, rmat);

	// normalize u
	v = arma::normalise(v) * sqrt(n);

	return 1;
}


int update_u_2(const arma::mat& A,
			const arma::mat& X,
			arma::colvec& u, 
			const arma::colvec& v, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l) 
{
	int n_obs = X.n_rows, k = X.n_cols, n = A.n_rows, n_c = n - n_obs;
	arma::colvec rmat = beta(k)*(y - X*beta.head(k) - beta(k+1)*v.head(n_obs))/n_obs + l*d*A.head_rows(n_obs)*v/n/n;
	u.head(n_obs) = rmat / (pow(beta(k),2)/n_obs + l*pow(d,2)/n);

	if(n_obs < n) {
		u.tail(n_c) = A.tail_rows(n_c)*v/d/n;
	}

	// normalize u
	u = arma::normalise(u) * sqrt(n);

	return 1;
}

int update_v_2(const arma::mat& A,
			const arma::mat& X,
			const arma::colvec& u, 
			arma::colvec& v, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l) 
{
	int n_obs = X.n_rows, k = X.n_cols, n = A.n_rows, n_c = n - n_obs;
	arma::colvec rmat = beta(k+1)*(y - X*beta.head(k) - beta(k)*u.head(n_obs))/n_obs + l*d*A.head_cols(n_obs).t()*u/n/n;
	v.head(n_obs) = rmat / (pow(beta(k+1),2)/n_obs + l*pow(d,2)/n);

	if(n_obs < n) {
		v.tail(n_c) = A.tail_cols(n_c).t()*u/d/n;
	}

	// normalize u
	v = arma::normalise(v) * sqrt(n);

	return 1;
}

int update_u_sym(const arma::mat& A,
			const arma::mat& X,
			arma::colvec& u, 
			const arma::colvec& y, 
			const arma::colvec& beta,
			double d,
			double l) 
{
	int n_obs = X.n_rows, k = X.n_cols, n = A.n_rows, n_c = n - n_obs;
	arma::mat lmat = -2*l*d*A/n;
	lmat.diag() += pow(beta(k),2) + l*pow(d,2);
	arma::colvec rmat = beta(k)*(y - X*beta.head(k));
	u = solve(lmat, rmat);
	
	// if(n_obs < n) {
	// 	u.tail(n_c) = A.tail_rows(n_c)*v/d/n;
	// }

	// normalize u
	u = arma::normalise(u) * sqrt(n);

	return 1;
}

bool stop_condition(const arma::colvec& u, 
					const arma::colvec& u_old,
					const arma::colvec& v, 
					const arma::colvec& v_old,
					double tol)
{
	int N = u.n_elem;

	arma::mat diff_u = (u_old * u_old.t() - u * u.t())/N;
	arma::mat diff_v = (v_old * v_old.t() - v * v.t())/N;

	return ((norm(diff_u, 2) > tol) | (norm(diff_v, 2) > tol));
}

// [[Rcpp::export]]
Rcpp::List predict_uv_R(const arma::mat& A,
			const arma::colvec& u_train, 
			const arma::colvec& v_train,
			arma::colvec u_test,
			arma::colvec v_test,
			double tol = 1e-4,
			int max_iter = 200)
{

	// arma::colvec u_test, v_test;

	int iter = predict_uv(A, u_train, v_train, u_test, v_test, tol, max_iter);

	return Rcpp::List::create(_("iter") = iter,
								_("u_test") = u_test,
							  _("v_test") = v_test);
}

int predict_uv(const arma::mat& A,
			const arma::colvec& u_train, 
			const arma::colvec& v_train,
			arma::colvec& u_test,
			arma::colvec& v_test,
			double tol,
			int max_iter)
{
	int N = A.n_rows, scaled = 1, n_train = u_train.n_elem, n_new = A.n_rows - u_train.n_elem;

	// if (norm(u_train, "fro") == 1) {scaled = 0;}

	// output
	arma::colvec u(N), v(N);
	double d = 0;

	// aux
	bool cond = 1; 
	arma::mat U, V;
	arma::vec s, tmp;
	arma::colvec u_old, v_old;
	double d_old = 0;
	int iter = 0;
	
	// initiate u, v
	// arma::svd(U, s, V, A);
	// double u_train_norm = norm(U(0, 0, size(u_train)));
	// double v_train_norm = norm(V(0, 0, size(u_train)));
	// u = U.col(0)/u_train_norm*sqrt(n_train); 
	// v = V.col(0)/v_train_norm*sqrt(n_train); 
	// d = s(0)/(u_train_norm * v_train_norm * n_train);
	// u.zeros(); v.zeros();
	u.subvec(0, size(u_train)) = u_train;
	v.subvec(0, size(v_train)) = v_train;
	u.subvec(n_train, size(u_test)) = u_test;
	v.subvec(n_train, size(v_test)) = v_test;

	// u = arma::normalise(u);
	// v = arma::normalise(v);

	while(cond & (iter < max_iter)) {

		u_old = u.tail(n_new); v_old = v.tail(n_new); d_old = d;

		// update d
		d = as_scalar(u.t() * A * v) / pow(norm(u, "fro"), 2) / pow(norm(v, "fro"), 2);
		// update u
		tmp = A.tail_rows(n_new) * v / d / pow(norm(v, "fro"), 2);
		u.tail(n_new) = tmp;
		// std::cout << tmp << std::endl;
		// u = arma::normalise(u);
		// update v
		tmp = A.tail_cols(n_new).t() * u / d / pow(norm(u, "fro"), 2);
		v.tail(n_new) = tmp;
		// std::cout << tmp << std::endl;
		// v = arma::normalise(v);
		
		cond = stop_condition(u.tail(n_new), u_old, v.tail(n_new), v_old, tol);

		iter++;
	}

	// if(scaled) {u *= sqrt(N); v *= sqrt(N); }
	
	u_test = u.tail(n_new);
	v_test = v.tail(n_new);

	return iter;
}

arma::colvec predict(const arma::mat& A,
			const arma::mat& X_test,
			const arma::colvec& beta,
			arma::colvec& u_test,
			arma::colvec& v_test)
{
	int n_new = X_test.n_rows, n_train = A.n_rows - X_test.n_rows;
	
	// output
	arma::mat U, V;
	arma::vec s;
	
	// initiate u, v
	irlba(U, s, V, A, 1);

	double u_train_norm = norm(U.col(0).head(n_train), "fro");
	double v_train_norm = norm(V.col(0).head(n_train), "fro");
	
	u_test = U.col(0).tail(n_new)/u_train_norm*sqrt(n_train); 
	v_test = V.col(0).tail(n_new)/v_train_norm*sqrt(n_train); 

	// u_test = U.col(0).tail(n_new)*sqrt(N);
	// v_test = V.col(0).tail(n_new)*sqrt(N);

	return min(join_rows(X_test, u_test, v_test) * beta, join_rows(X_test, -u_test, -v_test) * beta);
}

arma::colvec predict(const arma::mat& A,
			const arma::mat& X_test,
			const arma::colvec& u_train, 
			const arma::colvec& v_train,
			const arma::colvec& beta,
			arma::colvec& u_test,
			arma::colvec& v_test,
			double tol,
			int max_iter)
{
	
	predict_uv(A, u_train, v_train, u_test, v_test, tol, max_iter);

	return join_rows(X_test, u_test, v_test) * beta;

}

// [[Rcpp::export]]
Rcpp::List two_stage_c(const arma::mat& A,
			const arma::mat& X,
			const arma::colvec& y,
			double tol,
			int max_iter)
{
	int N = A.n_rows;

	// output
	arma::colvec beta, u, v, residuals, u_distance(max_iter);
	double d = 0;

	// aux
	bool cond = 1; 
	arma::mat U, V;
	arma::vec s, tmp;
	arma::colvec u_old, v_old;
	double d_old = 0;
	int iter = 0;
	
	// initiate u, v
	arma::svd(U, s, V, A);
	// u = U.col(0)*sqrt(N); v = V.col(0)*sqrt(N);	
	d = s(0)/N;
	u.randu(N); v.randu(N);
	u = arma::normalise(u) * sqrt(N);
	v = arma::normalise(v) * sqrt(N);
	
	while(cond & (iter < max_iter)) {

		u_old = u; v_old = v; d_old = d;
		// update u
		u =  A * v / d / N;
		u = arma::normalise(u) * sqrt(N);
		// update v
		v = A.t() * u / d / N;
		v = arma::normalise(v) * sqrt(N);
		// update d
		d = as_scalar(u.t() * A * v) / pow(N, 2);

		u_distance(iter) = norm(u_old * u_old.t() - u * u.t(), 2);
		cond = stop_condition(u, u_old, v, v_old, tol);

		iter++;
	}
	
	update_beta(X, u, v, y, beta);

	residuals = y - join_rows(X, u, v) * beta;

	Rcpp::StringVector method = "two_stage_c";

	return Rcpp::List::create(_("u") = u,
							  _("v") = v,
							  _("beta") = beta,
							  _("d") = d,
							  _("iter") = iter,
							  _("residuals") = residuals,
							  _("method") = method,
							  _("max_iter") = max_iter,
							  _("u_distance") = u_distance.head(iter));
}



int irlba (
	arma::mat& U, 
	arma::vec& s, 
	arma::mat& V,  
	const arma::mat& A,
	int k)
{

	// Obtain environment containing function
	Rcpp::Environment base("package:irlba"); 

	// Make function callable from C++
	Rcpp::Function svd_r = base["irlba"];    

	// Call the function and receive its list output
	Rcpp::List res = svd_r(Rcpp::_["A"] = A,
	                       Rcpp::_["nu"]  = k,
	                       Rcpp::_["nv"]  = k); 

	U = as<arma::mat>(res["u"]);
	V = as<arma::mat>(res["v"]);
	s = as<arma::vec>(res["d"]);

	return res["iter"];
}


int irlba_eigen (
	arma::mat& U, 
	arma::vec& s, 
	const arma::mat& A,
	int k)
{

	// Obtain environment containing function
	Rcpp::Environment base("package:irlba"); 

	// Make function callable from C++
	Rcpp::Function eigen_r = base["partial_eigen"];

	// Call the function and receive its list output
	Rcpp::List res = eigen_r(Rcpp::_["x"] = A,
	                       Rcpp::_["n"]  = k); 

	U = as<arma::mat>(res["vectors"]);
	s = as<arma::vec>(res["values"]);

	return 1;
}



// [[Rcpp::export]]
Rcpp::List test_rirlba (const arma::mat& A)
{

	arma::mat U, V;
	arma::vec s;

	irlba(U, s, V, A, 1);

	return Rcpp::List::create(	_("s") = s, 
								_("u") = U,
							 	_("v") = V);
}



// // [[Rcpp::export]]
// arma::colvec kron_diag (const arma::mat& A, const arma::mat& B) {

// 	int n_a = A.n_rows, n_b = B.n_rows;
// 	arma::colvec diag_ret(n_a * n_b);

// 	for(int i = 0; i < n_a; i++) {
// 		for(int j = 0; j < n_b; j++) {

// 		}
// 	}

// 	diag_ret;
// }