#include<iostream>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd function_y(MatrixXd para, MatrixXd x);
MatrixXd getJacobianMatrix(MatrixXd para_est, MatrixXd deflection, MatrixXd ym, MatrixXd input);
MatrixXd LevenbergMarquardtFit(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, MatrixXd input);
