#ifndef AdvectionDG_H
#define AdvectionDG_H

//deal.ii includes
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

    // Distributed specific
#include <deal.II/distributed/solution_transfer.h>

    // MeshWorker specific
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/vector_selector.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

// c++ includes
#include <fstream>
#include <iostream>
#include <sstream>

// local includes
//#include "../base/post_processor.h"
#include "vector_space.h"
#include "fsi_utilities.h"
#include "boundary_conditions.h"
#include "boundary_indicators.h"
#include "explicit_time_discretization.h"
#include "user_defined_constants.h"

    //! Template function that can be used to suppress compiler warnings
template<class T> void unused( const T& ) { }

//Namespaces we want to use
using namespace dealii;
    //using namespace dealii::Functions;

// Data storage for use in `applyLimiters' which uses the Zhang&Shu scaling
// about the cell average.  Ideas from Liu&Osher for the TVB limiter that is
// used on pressure & velocity.
template<int dim>
class limiter_data
{
public:
    limiter_data(){*this = 0.0;}
    ~limiter_data(){}
    
    double u_min, u_avg, u_max;
    
    void operator= (const double &value){
        u_avg = value;
        u_min = value;
        u_max = value;
    }
    
    void set_default(){
            // Wildly out of range values for mins and maxes,
            // averages set to zero
        u_avg = 0.0;
        u_min = 1.e+16;
        u_max = -1.e+16;
    }
};//class limiter_data

    //! For our stabilization operators, we need to have certain data for each cell.  
    //! We will store them in this class:
struct userCellData
{
    double G_k_i, g_k_i, h_k, volK, volK_34;
    
    int material_id;
    
        // Copy operator
    void operator= (const userCellData& ucd)
    {
        G_k_i = ucd.G_k_i;
        g_k_i = ucd.g_k_i;
        h_k   = ucd.h_k;
        volK  = ucd.volK;
        volK_34 = ucd.volK_34;
        
        material_id = ucd.material_id;
    }//=
    
        // Initialize, typically value = 0.0
    void operator= (const double& value)
    {
        G_k_i = value;
        g_k_i = value;
//        h_k   = value;
//        volK  = value;
//        volK_34 = value;
    }//=
};


/** Two phase DG class **/
template <int dim>
class AdvectionDG
{
public:
    
    AdvectionDG ();
    
    ~AdvectionDG ();
    
    void run (std::string data_out_path);
    
    void reinit(VectorSpace<dim> &vspace);
    
    /** Parse the parameter file. */
    void parse_parameters(ParameterHandler &prm);

    /** Generate entries in the given parameter file. */
    void declare_parameters(ParameterHandler &prm);
    
    struct Flux{
        double u;
        void print(){
            std::cout<<"\nf.u = "<<u;
        };
    };
    
    struct FluxJacobian{
            // flux-jacobian of alpha equation
        double dF_du;
        
        void operator *=(const double& factor)
            {
                dF_du *= factor;
            }//overloaded *= operator
        
        void operator += (const FluxJacobian& fj)
        {
            dF_du += fj.dF_du;
        }//overloaded += operator
        
        void operator = (const FluxJacobian& fj)
        {
            dF_du = fj.dF_du;
        }//overloaded = operator
        
    };//FluxJacobian

private:
    
        //! Conditional output; only write to std::out on the `head' node
    ConditionalOStream                pcout;
    
    bool assemble_jacobian;
    double initial_residual_norm;
    
    typedef MeshWorker::DoFInfo<dim> DoFInfo;
    typedef MeshWorker::IntegrationInfo<dim> CellInfo;
    typedef FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> CellFilter;
    
//    Threads::ThreadMutex resource_lock;
    
    void setup_system ();
    
    void timestep_reinit ();
    
    void projectInitialConditions ();
    
    void projectInitialConditions_interval (const typename DoFHandler<dim>::active_cell_iterator &begin,
                                            const typename DoFHandler<dim>::active_cell_iterator &endc);
    
    void applyLimiters (ParVec& solution,
                        ParVec& old_solution);
    
    void update_limiter_data (ParVec& solution,
                              ParVec& old_solution,
                              bool update_geometry_info=true);
    
    void cell_laplacian_limiter (ParVec& solution);
    
    bool is_troubled (const typename DoFHandler<dim>::active_cell_iterator cell);

    
        //! Call this from run after td.advance.  Takes care of any solution "tweaking", etc, that we
        //! might want to do.
    void finalize_timestep ();
    
   
        //! Function for TimeDiscretization
    void assemble_rhs (ParVec& solution,
                       ParVec& residual,
                       bool assemble_jacobian=true);
    
    void assemble_mass_matrix_and_multiply (ParVec& cur_solution,
                                              ParVec& residual);
    
    bool check_convergence (unsigned int iteration, 
                            ParVec& residual);
    
    void init_newton_iteration (int stage);
    
        //! Functions for MeshWorker
    void integrate_mass(ParVec& solution,
                        ParVec& residual,
                        const typename DoFHandler<dim>::active_cell_iterator &begin,
                        const typename DoFHandler<dim>::active_cell_iterator &endc);
    
    void integrate_cell_term (DoFInfo& dinfo,
                                     CellInfo& info);
    void integrate_boundary_term (DoFInfo& dinfo,
                                         CellInfo& info);
    void integrate_face_term (DoFInfo& dinfo1,
                                     DoFInfo& dinfo2,
                                     CellInfo& info1,
                                     CellInfo& info2);
   
    void solve ();
    
    void refine_grid (int max_level);
    void refine_notional_cav_grid (double ymax);
    
    void output_results (int timestep, std::string data_out_path);
    
    inline unsigned int get_boundary_index (unsigned int boundary_id) const;
    
    /** compute parameters for stabilization,
     *  set the cell user_pointer
     *  if on the first timestep
     */
    void update_cell_data (bool update_geometry_info=true);
    
    void compute_avg_min_max (DoFInfo& dinfo,
                              CellInfo& info);
    
    double compute_u_2d(double time, Point<dim> &point);
    
    void integrate_boundary_discontinuity_indicator (DoFInfo& dinfo,
                                                     CellInfo& info);
    
    void integrate_face_discontinuity_indicator (DoFInfo& dinfo1,
                                            DoFInfo& dinfo2,
                                            CellInfo& info1,
                                            CellInfo& info2);

    SmartPointer<VectorSpace<dim> > vspace;

    SparsityPattern             sparsity_pattern;
    SparseMatrix<double>        stiffness;
    SparseMatrix<double>        nonlinear_mass;
    SparseMatrix<double>        linear_mass;
    
    std::vector<userCellData>   cell_data;
    std::vector<limiter_data<dim> >   cell_limiter_data;
    
    std::vector<std::pair<double,std::string> > times_and_names;

        //    TimeDiscretization<dim> td;
    ExplicitTimeDiscretization<dim> td;
   
    double              h_max;
    double              h_min;
    
    double diffusion_power;
    
    //! System component information
    static const unsigned int u_comp = 0;
    static const unsigned int n_comp = 1;
    
    //! Maximum number of domain boundaries
    static const unsigned int max_n_boundaries = 10;
    
    const FEValuesExtractors::Scalar U;
    
        //! HacK:  I need to know the current time in many places!
    double cur_time;
    
    
    //! Initial conditions function
    FunctionParser<dim> initial_conditions;
    
    std::vector<BoundaryConditions<dim> > boundary_conditions;
    
    int output_freq, refine_freq, refine_levels, ic_refinements;
    
        //! First stab at any sort of restart capability:
    void writeSolutionToDisk (int timestep, std::string data_out_path) const;
    void readSolutionFromDisk (std::string data_out_path);

        //! Name for the case we want to run.  Specified in prm file.
    std::string caseName;
    
    int fileNum;
    
        // Flag to decide whether or not to read the starting solution from disk
    bool restart;
};


#endif
