//
//  explicit_time_discretization.h
//  
//
//  Created by Scott Miller on 10/24/12.
//   - improve efficiency and data management over "TimeDiscretization" 
//     for explicit time stepping methods
//
//  Copyright 2012 Applied Research Lab, Penn State. All rights reserved.
//

#ifndef _exp_time_discretization_h
#define _exp_time_discretization_h

#include <iostream>

#include <base/subscriptor.h>
#include <base/std_cxx1x/function.h>
#include <base/parameter_handler.h>
#include <deal.II/base/conditional_ostream.h>

#include <lac/vector.h>
#include <lac/compressed_sparsity_pattern.h>
#include <lac/sparse_matrix.h>
#include <lac/parallel_vector.h>
#include <lac/solver_gmres.h>
#include <lac/sparse_ilu.h>
#include <lac/sparse_direct.h>
#include <lac/precondition.h>

    // Local includes
#include "vector_space.h"
#include "user_defined_constants.h"

    //! Define a parallel vector type so I can switch
    //! if need be.
typedef dealii::parallel::distributed::Vector<double> ParVec;


template <int dim>
class ExplicitTimeDiscretization : public dealii::Subscriptor
{
public:
    enum DiscretizationType {
        BackwardEuler,
        RK1,
        RK2,
        RK3,
        RK4,
        RK5,
        BDF2,
        SSP_5_4};
    
private:
    
        // Vector Space -- for DOF info
    SmartPointer<VectorSpace<dim> > vspace;
    
    bool implicit; // implicit? or explicit
    
        //! Number of timesteps 
    unsigned int n_timestep;
    
        //! Number of RK stages
    unsigned int n_stages;
    
        //! Set the current time
    double cur_time;
    
        //! Final time of the simulation
    double final_time;
    
        //! Time scheme to use.
        //! Integer refers to enumeration list
    DiscretizationType time_scheme;
    
    
        //! Store a few previous time step sizes, depending
        //! on the time discretization we use.
        //! Necessary for non-constant time step sizes.
    std::vector<double> prev_delta_t;
    
        //! Is the mesh topology changing?
        //! Currently this class only deals well when the value == false
        //! If mesh_changed==true, we shall throw an error of ExcNotImplemented
    bool mesh_changed;
    
        //! Number of dofs.  Constant in the current incarnation, as mesh_changed=false
    unsigned int n_dofs;
    
        //! Solution vectors, current through N-previous.
    //! N=1 for single step or RK methods
    //! N>1 for linear multistep methods
    std::vector<ParVec>         solution;
    
        //! Vectors for stages in RK methods
    std::vector<ParVec>        stages;
    
        //! For SSP methods I need more vectors!
    std::vector<ParVec>        u_i;
    
        //! Number of previous time steps to keep
    unsigned int num_old_times;
    
        //! We store some functionals to compute the RHS and check convergence:
        //! We set these in the constructor.
        //! These functions are provided by the `spatial discretization'
    dealii::std_cxx1x::function<void (ParVec&, ParVec&,
                                    bool)>
                        assemble_rhs;
    
    dealii::std_cxx1x::function<void (ParVec&, ParVec&)> 
                        assemble_mass_matrix_and_multiply;
    
    dealii::std_cxx1x::function<bool (unsigned int, ParVec&)> 
                        check_convergence;
    
    dealii::std_cxx1x::function<void (int)>            
                        init_newton_iteration;
    
    dealii::std_cxx1x::function<void (ParVec&, ParVec&)>
                        applyLimiters;
    
    
        //! Internal pointers
    ParVec* current_solution;
    ParVec* old_solution;
    
//    dealii::Vector<double> newton_update, residual, mass_residual;
    
    void set_time_scheme (std::string scheme);
    
    //! Initialize functions; set up necessary data structures
    void initialize (void);
    
        //! advance functions for specific schemes
    void advance_RK1 (void);
    void advance_RK2 (void);
    void advance_RK3 (void);
    void advance_RK4 (void);
    void advance_RK5 (void);
    void advance_SSP_5_4 (void);
    
    void compute_residual (unsigned int u_i_index, unsigned int stage_index);
    
        //! Output only on head proc
    ConditionalOStream pcout;
    
public:
    
        //! Constructor
    ExplicitTimeDiscretization (//MPI_Comm& mpi_communicator,
                        const dealii::std_cxx1x::function<void (ParVec&, ParVec&,
                                                           bool)>                    
                                                    &assemble_rhs,
                        const dealii::std_cxx1x::function<void (ParVec&, ParVec&)> 
                                                    &assemble_mass_matrix_and_multiply,
                        const dealii::std_cxx1x::function<bool (unsigned int, ParVec&)> 
                                                    &check_convergence,
                        const dealii::std_cxx1x::function<void (int)>                    
                                                    &init_newton_iteration,
                        const dealii::std_cxx1x::function<void (ParVec&, ParVec&)>
                                                    &applyLimiters);
    
        //! Destructor
    ~ExplicitTimeDiscretization ();
    
        //! Set pointer to the vector space:
    inline void set_vector_space(VectorSpace<dim> &vs){vspace = &vs;};
    
    /** Reinit. */
    void reinit(unsigned int n_dofs, double time=-1.); 
    
    /** Ghost cell update for solution & old_solution */
    void update_ghost_cells (void)
        {
        current_solution->update_ghost_values();
        old_solution->update_ghost_values();
        };
    
    /** Read time discretization parameters */
    void parse_parameters(dealii::ParameterHandler &prm);
    
    /** Generate entries in the given parameter file. */
    void declare_parameters(dealii::ParameterHandler &prm);
    
    
        //! Query timestep size
    inline double time_step_size (void){return delta_t;};
    
        //! Query timestep number
    inline unsigned int time_step_num (void){return n_timestep;};
    
        //! Access current time
    inline double current_time (void){return cur_time;};
    
        //! Set flag for changing topology
    inline void set_topology_changed (bool flag) {mesh_changed = flag;};
    
        //! Access solution vector used to compute the spatial discretization
        //! This would be the t^{n+1} solution for fully implicit, 
        //! or the t^n solution for explicit (although for more general methods,
        //! we will need multiple "old" solutions... TODO:  add this capability
    ParVec& access_current_solution (void){return *current_solution;};
    
    const ParVec& access_current_solution (void) const {return *current_solution;};
    
        //! Soley for setting initial conditions
    ParVec& access_old_solution (void){return *old_solution;};
    
    void set_current_from_old (void){*(this->current_solution) = *(this->old_solution);};
    void set_old_from_current (void){*(this->old_solution) = *(this->current_solution);};
    
        //! Are we at the final time yet?
    inline bool finalized (void) 
    {
        if (cur_time < final_time)
            return false;
        
        return true;
    }//finalized
    
        //! Advance one time step
    void advance (void);
    
        //! Current timestep size
    double delta_t;
    
};//class-TimeDiscretization


#endif
