#ifndef BASE_H
#define BASE_H

//deal.ii includes
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/vector.h>

#include <deal.II/hp/dof_handler.h>

//C++ includes
#include <fstream>
#include <iostream>

//Program Specific Includes
#include "domain.h"
#include "vector_space.h"
#include "AdvectionDG.h"
#include "fsi_utilities.h"  

//namespace usage
using namespace dealii;
using namespace dealii::Functions;


/**
* The base driver class for the entire problem
*/

//define the class, and make it use a template
template <int dim>
class Base 
{
public:
    /** Constructor
    * Currently does nothing extra. */
        Base ();
    /** Destructor.
    Currently does nothing extra. */
        ~Base ();
    /** Runs the elastic problem, given different input parameters.
    Currently set to run a static and dynamic case, each with nonlinear consideration. */
    
    //the public function called from main.cc
    void run (std::string &prm_file_name);

private:

    //! Distributed computations:
    MPI_Comm mpi_communicator; 

    /* These classes are needed for almost all problems */
    Domain<dim>          domain;
    VectorSpace<dim>     vspace;

//    ElasticityDG<dim> elastic;
//    TwoPhaseDG<dim> navier;
    AdvectionDG<dim> advection;
    
    ConditionalOStream pcout;

    /* Items that are independent of the problem
       being solved; they are instrinsic to Base<dim> */

    //Parameter Handler
    ParameterHandler     prm;
    void declare_parameters ();
    void parse_parameters ();

	//some functions	
    void run_cc(unsigned int cc); //cc stands for convergence cycles
    void reinit_cc(unsigned int cc);

	//data writing stuff
//    void write_data(Vector<double> &u, unsigned int cc);
    std::string data_out_path;
    bool write_vtu;

	//convergence and error stuff
//    ErrorHandler<dim> error_handler;
//    Vector<double> error_solution;
//    ParsedFunction<dim> exact_solution;
    unsigned int number_of_cc;


};



#endif
