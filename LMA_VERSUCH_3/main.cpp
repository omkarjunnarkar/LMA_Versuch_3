#include<iostream>
#include<Eigen/Dense>
#include<iomanip>
#include<fstream>
#include<math.h>
#include<vector>
#include"src/rapidcsv.h"
#include"functions.h"

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

using namespace std;
using namespace Eigen;
using namespace rapidcsv;

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/

void main() {

	cout << "Main program started..." << endl;

	Document dx("xdata.csv");
	Document dy("y_measured.csv");

	vector<double> xcol = dx.GetColumn<double>(0);
	vector<double> ycol = dy.GetColumn<double>(0);
	
	int data_size = ycol.size();
	MatrixXd xdata(ycol.size(), 1);
	MatrixXd y_measured(ycol.size(), 1);

	for (int u = 0; u < data_size; u++) {
		xdata(u, 0) = xcol[u];
		y_measured(u, 0) = ycol[u];
	}

	MatrixXd parameters_guess{
		{-3.4,0.0,1.0}  //Initial Guess for parameters
	};

	MatrixXd initial_deflection{
		{1e-8,1e-8,1e-8}	//Initial deflection for parameters(to find Jacobian Matrix using finite difference scheme)
	};

	parameters_guess = parameters_guess.reshaped(parameters_guess.cols(), 1);
	initial_deflection = initial_deflection.reshaped(parameters_guess.rows(), 1);

	cout << "Data sourced, calling LMA. \n";

	MatrixXd Parameters = LevenbergMarquardtFit(parameters_guess, initial_deflection, y_measured, xdata);

	cout << "Parameters = \n" << Parameters << endl;
	//cout << "------***********-------------*******************----------------********************-------" << endl;

}

