//
//  explicit_time_discretization.templates.h
//  
//
//  Created by Scott Miller on 10/24/12.
//  Copyright 2012 Applied Research Lab, Penn State. All rights reserved.
//
#include "../include/explicit_time_discretization.h"

using namespace dealii;

/** Empty constructor. */
//template <int dim>
//TimeDiscretization<dim>::TimeDiscretization (const dealii::std_cxx1x::function<void (void)> &test)
//: n_timestep(0),
//  cur_time(-1.0)
//{
//        //    init_newton_iteration = test;
//}

/** Full constructor */
template <int dim>
ExplicitTimeDiscretization<dim>::ExplicitTimeDiscretization (
//         MPI_Comm& mpi_communicator,
         const dealii::std_cxx1x::function<void (ParVec&, ParVec&, bool)>                    
                                        &assemble_rhs,
         const dealii::std_cxx1x::function<void (ParVec&, ParVec&)> 
                                        &assemble_mass_matrix_and_multiply,
         const dealii::std_cxx1x::function<bool (unsigned int, ParVec&)> 
                                        &check_convergence,
         const dealii::std_cxx1x::function<void (int)>                  
                                        &init_newton_iteration,
         const dealii::std_cxx1x::function<void (ParVec&, ParVec&)>
                                        &applyLimiters)
: n_timestep(0),
  cur_time(-1.0),
  pcout (std::cout,
       Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    this->assemble_rhs = assemble_rhs;
    this->assemble_mass_matrix_and_multiply = assemble_mass_matrix_and_multiply;
    this->check_convergence = check_convergence;
    this->init_newton_iteration = init_newton_iteration;
    this->applyLimiters = applyLimiters;
}//constructor

/** Destructor. */
template <int dim>
ExplicitTimeDiscretization<dim>::~ExplicitTimeDiscretization ()
{}

/** Reinit. */
template <int dim>
void ExplicitTimeDiscretization<dim>::reinit(unsigned int dofs, double time)
{
    n_dofs = dofs;
    
    pcout<<"\nReinitializing "<<n_stages<<"-stage Explicit Time Integrator with n_dofs = "<<n_dofs<<"\n";
    
    if (time < -0.50)
        initialize();
    else
    {
        cur_time = time;
            // Reinitialize data for new # of dofs, but don't clear the solution vector!
        for(unsigned int i=0; i<stages.size(); ++i)
        {
            stages[i].reinit(vspace->locally_owned_dofs,
                              vspace->locally_relevant_dofs,
                             vspace->mpi_communicator);
            
            u_i[i].reinit(vspace->locally_owned_dofs,
                             vspace->locally_relevant_dofs,
                             vspace->mpi_communicator);
        }
    }
    
}//reinit

/** set_time_scheme. */
template <int dim>
void ExplicitTimeDiscretization<dim>::set_time_scheme(std::string scheme)
{
    if (scheme == "RK1")
        {
            time_scheme = RK1;
            n_stages = 1;
        }  
    else if (scheme == "RK2")
        {
            time_scheme = RK2;
            n_stages = 2;
        }
    else if (scheme == "RK3")
        {
            time_scheme = RK3;
            n_stages = 3;
        }
    else if (scheme == "RK4")
        {
            time_scheme = RK4;
            n_stages = 4;
        }
    else if (scheme == "RK5")
        {
            time_scheme = RK5;
            n_stages = 5;
        }
    else if (scheme == "SSP_5_4")
        {
            time_scheme = SSP_5_4;
            n_stages = 5;
        }
    else
        {
        pcout<< sc::warning << sc::nl 
            << "TIME SCHEME = "<< scheme 
            << " NOT FOUND" << sc::nl;
        
        Assert(false, ExcNotImplemented());           
        }

}//set_time_scheme

/** initialize()
 *  - size vectors using info obtained
 *    in set_time_scheme, etc
 */
template <int dim>
void ExplicitTimeDiscretization<dim>::initialize ()
{
    pcout<<"\nInitializing "<<n_stages<<"-stage Explicit Time Integrator with n_dofs = "<<n_dofs<<std::endl;
        
        // this class is all explicit schemes
    implicit = false;
    
        //! Need 2 time levels/solution vectors
        //! Hence, only 1 old time
    num_old_times = 1;
    
    ParVec vec(vspace->locally_owned_dofs,
               vspace->locally_relevant_dofs,
               vspace->mpi_communicator);
    
    vec = 0.0;
    
    solution.push_back(vec);//current_solution
    solution.push_back(vec);//old_solution
    
    current_solution = &(solution[0]);
    old_solution     = &(solution[1]);
    
        //! Vectors for k_i stages
    for(unsigned int ns=0; ns<=n_stages; ++ns)
    {
        stages.push_back(vec);
        u_i.push_back(vec);
    }
    
}//initialize

/** Read mesh file name, etc. */
template <int dim>
void ExplicitTimeDiscretization<dim>::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time Discretization");
    
    double init_time = prm.get_double("Initial time");
    
    if (cur_time < 0.0)
        cur_time = init_time;
    
    final_time = prm.get_double("Final time");
    
    delta_t = prm.get_double("Timestep size");
    
    std::string scheme = prm.get("Time scheme");

    prm.leave_subsection();
      
    set_time_scheme(scheme);
}//parse_parameters

/** Generate entries in the given parameter file. */
template <int dim>
void ExplicitTimeDiscretization<dim>::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time Discretization");
    
    prm.declare_entry ("Initial time", "0.0", Patterns::Double(),
                       "Starting time for the simulation.");
    prm.declare_entry ("Final time", "1.0", Patterns::Double(),
                       "Final time for the simulation.");
    prm.declare_entry ("Timestep size", "0.001", Patterns::Double(),
                       "Size of the timestep to be used.");
    prm.declare_entry("Time scheme",
                      "RK1",
                      Patterns::Selection("RK1|RK2|RK3|RK4|RK5|SSP_5_4"),
                                         "<RK1|RK2|RK3|RK4|RK5|SSP_5_4>");
    prm.leave_subsection();
}//declare_paramaters


template <int dim>
void ExplicitTimeDiscretization<dim>::compute_residual (unsigned int u_i_index, unsigned int stage_index)
{
        //! Make call to SD so it can do what it needs to 
        //! for each newton iteration/explicit stage
    init_newton_iteration(stage_index);
    
    stages[stage_index] = 0.0;
    
    assemble_rhs(u_i[u_i_index],
                 stages[stage_index],
                 false);
    
        // The RHS is constant, but I need to do Newton iterations
        // here because the 'mass' matrix is solution dependent.
        // This ensures element-wise balance to machine precision.
    assemble_mass_matrix_and_multiply(u_i[u_i_index],
                                      stages[stage_index]);
    
}//end-compute_residual


template <int dim>
void ExplicitTimeDiscretization<dim>::advance(void)
{
    if (time_scheme == RK1)
        return advance_RK1();
    else if (time_scheme == RK2)
        return advance_RK2();
    else if (time_scheme == RK3)
        return advance_RK3();
    else if (time_scheme == RK4)
        return advance_RK4();
    else if (time_scheme == SSP_5_4)
        return advance_SSP_5_4();
    else
    {
        pcout<< sc::warning << sc::nl 
        << "advance() not implemented for time scheme = "<< time_scheme << sc::nl;
        
        Assert(false, ExcNotImplemented());
    }
        
}//advance


/**********************************************
 * RK1 := Explicit Euler
 *
 *  y^{n+1} = y^{n} + f(t^{n}, y^{n})
 *
 **********************************************/
template <int dim>
void ExplicitTimeDiscretization<dim>::advance_RK1(void)
{
    n_timestep++;
    
    double old_time = cur_time;
    
    pcout << sc::nl << "Timestep: " << n_timestep << sc::nl
            << "Global time = " << cur_time << sc::nl;
    
//    applyLimiters(*(this->current_solution));
    current_solution->update_ghost_values();
    
        //! Set the old_solution equation to the current
        //! The "current" solution is then the initial guess
        //! for the Newton iterations
    set_old_from_current();
    
        //! Create temp vector
    ParVec &tmp_vec = stages[1];
    
    u_i[0] = *(this->current_solution);
    u_i[0].update_ghost_values();
    
        //! Stage 1
    compute_residual(0,0);
//    stages[0].update_ghost_values();
    
        // Commented out the below 4 lines for mass conservation test
    tmp_vec = stages[0];
    tmp_vec *= (delta_t);
    u_i[1] = tmp_vec;
    u_i[1] += u_i[0];
    
        // Hack for mass conservation test
//    u_i[1] = u_i[0];
    
        //    stages[0].update_ghost_values();
    u_i[1].update_ghost_values();
    
    applyLimiters(u_i[1], u_i[0]);
    
        //! Final solution for the timestep
    *(this->current_solution) = u_i[1];
//    current_solution->update_ghost_values();
    
        //! Note:  the code below counts ghost cells twice!  
    const double locally_relevant_norm = u_i[1].l2_norm();
    
    const double total_norm = std::sqrt(Utilities::MPI::sum (locally_relevant_norm, MPI_COMM_WORLD));
    
    pcout<<"\nl2 norm of solution = "<<total_norm<<std::endl;
    
    cur_time = old_time + delta_t;

    
}

template <int dim>
void ExplicitTimeDiscretization<dim>::advance_RK2(void)
{
//    n_timestep++;
//    
//    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
//                        << "Global time = "<< cur_time << sc::nl;
//    
//        //! Set the old_solution equation to the current
//        //! The "current" solution is then the initial guess
//        //! for the Newton iterations
//    set_old_from_current();
//    
//        //! Create temp vector
//    Vector<double> &tmp_vec = stages[2];
//    tmp_vec = *(this->current_solution);
//    
//        //! Stage 1
//    NewtonRaphson_RK(false);
//    stages[0] = *(this->current_solution);
//        // current_solution is really k1, adjust accordingly
//    *(this->current_solution) *= (0.5*delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//        //! Update time for stages 2 and 3
//    cur_time += 0.5*delta_t;
//    
//        //! Stage 2
//    NewtonRaphson_RK(false);
//    stages[1] = *(this->current_solution);
//    
//    cur_time += 0.5*delta_t;
//    
//        //! Set final solution
//    residual = stages[1];
//    residual *= delta_t;
//    set_current_from_old();
//    *(this->current_solution) += residual;
}

template <int dim>
void ExplicitTimeDiscretization<dim>::advance_RK3(void)
{
//    n_timestep++;
//    
//    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
//    << "Global time = "<< cur_time << sc::nl;
//    
//        //! Set the old_solution equation to the current
//        //! The "current" solution is then the initial guess
//        //! for the Newton iterations
//    set_old_from_current();
//    
//        //! Create temp vector
//    Vector<double> &tmp_vec = stages[3];
//    tmp_vec = *(this->current_solution);
//    
//        //! Stage 1
//    NewtonRaphson_RK(false);
//    stages[0] = *(this->current_solution);
//        // current_solution is really k1, adjust accordingly
//    *(this->current_solution) *= (0.5*delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 1 finished"<<sc::nl;
//    
//        //! Update time for stage 2
//    cur_time += 0.5*delta_t;
//    
//        //! Stage 2
//    NewtonRaphson_RK(false);
//    stages[1] = *(this->current_solution);
//        // current_solution is really k2, adjust accordingly
//    *(this->current_solution) *= (2.0*delta_t);
//    tmp_vec = stages[0];
//    tmp_vec *= -2.0;
//    tmp_vec += *(this->old_solution);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 2 finished"<<sc::nl;
//    
//        //! Stage 3
//        //! Update time for stage 3
//    cur_time += 0.5*delta_t;
//    NewtonRaphson_RK(false);
//    stages[2] = *(this->current_solution);
//    
//    std::cout<<"Stage 3 finished"<<sc::nl;
//    
//        //! Compute the solution at the end of the timestep
//        // abuse my residual vector a little bit:
//    residual = stages[1];
//    residual *= 4.0;
//    residual += stages[0];
//    residual += stages[2];
//    residual *= (delta_t/6.0);
//    set_current_from_old();
//    *(this->current_solution) += residual;
}

template <int dim>
void ExplicitTimeDiscretization<dim>::advance_RK4(void)
{
//    n_timestep++;
//    
//    std::cout << sc::nl << "Timestep: " << n_timestep << sc::nl
//    << "Global time = "<< cur_time << sc::nl;
//    
//        //! Set the old_solution equation to the current
//        //! The "current" solution is then the initial guess
//        //! for the Newton iterations
//    set_old_from_current();
//    
//        //! Create temp vector
//    Vector<double> &tmp_vec = stages[4];
//    tmp_vec = *(this->current_solution);
//    
//        //! Stage 1
//    NewtonRaphson_RK(false);
//    stages[0] = *(this->current_solution);
//        // current_solution is really k1, adjust accordingly
//    *(this->current_solution) *= (0.5*delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 1 finished"<<sc::nl;
//    
//        //! Update time for stages 2 and 3
//    cur_time += 0.5*delta_t;
//    
//        //! Stage 2
//    NewtonRaphson_RK(false);
//    stages[1] = *(this->current_solution);
//        // current_solution is really k2, adjust accordingly
//    *(this->current_solution) *= (0.5*delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 2 finished"<<sc::nl;
//    
//        //! Stage 3
//    NewtonRaphson_RK(false);
//    stages[2] = *(this->current_solution);
//        // current_solution is really k3, adjust accordingly
//    *(this->current_solution) *= (delta_t);
//    tmp_vec += *(this->current_solution);
//    *(this->current_solution) = tmp_vec;
//    
//    std::cout<<"Stage 3 finished"<<sc::nl;
//    
//        //! Update time for stage4
//    cur_time += 0.5*delta_t;
//    
//        //! Stage 4
//    NewtonRaphson_RK(false);
//    stages[3] = *(this->current_solution);
//    
//    std::cout<<"Stage 4 finished"<<sc::nl;
//    
//        //! Compute the solution at the end of the timestep
//        // abuse my residual vector a little bit:
//    residual = stages[1];
//    residual += stages[2];
//    residual *= 2.0;
//    residual += stages[0];
//    residual += stages[3];
//    residual *= (delta_t/6.0);
//    set_current_from_old();
//    *(this->current_solution) += residual;
//
}//advance_RK4

template <int dim>
void ExplicitTimeDiscretization<dim>::advance_RK5(void)
{
    n_timestep++;
    
    double old_time = cur_time;
    
    pcout << sc::nl << "Timestep: " << n_timestep << sc::nl
              << "Global time = " << cur_time << sc::nl;
    
//    applyLimiters(*(this->current_solution));
    
        //! Set the old_solution equation to the current
        //! The "current" solution is then the initial guess
        //! for the Newton iterations
    set_old_from_current();
    
        //! Create temp vector
    ParVec &tmp_vec = stages[5];
    
    u_i[0] = *(this->current_solution);
    
        //!  Note:  stages[j] = RHS(u_i[j])
    
        //! NON-SSP RK version
        //! Stage 1
    compute_residual(0,0);
    
    tmp_vec = stages[0];
    tmp_vec *= (delta_t)*(1./7.);
    u_i[1] = tmp_vec;
    u_i[1] += u_i[0];
    
//    applyLimiters(u_i[1]);
    
        //! Stage 2
    compute_residual(1,1);
    tmp_vec = stages[1];
    tmp_vec *= (delta_t)*(3./16.);
    u_i[2] = tmp_vec;
    tmp_vec = u_i[0];
    u_i[2] += tmp_vec;
    
//    applyLimiters(u_i[2]);
    
        //! Stage 3
    compute_residual(2,2);
    tmp_vec = stages[2];
    tmp_vec *= (delta_t)*(1./3.);
    u_i[3] = tmp_vec;
    tmp_vec = u_i[0];
    u_i[3] += tmp_vec;
    
//    applyLimiters(u_i[3]);
    
        //! Stage 4
    compute_residual(3,3);
    tmp_vec = stages[3];
    tmp_vec *= (delta_t)*(2./3.);
    u_i[4] = tmp_vec;
    tmp_vec = u_i[0];
    u_i[4] += tmp_vec;
    
//    applyLimiters(u_i[4]);
    
        //! Stage 5
    compute_residual(4,4);
    tmp_vec = stages[4];
    tmp_vec *= (delta_t)*(3./4.);
    u_i[5] = tmp_vec;
    tmp_vec = stages[0];
    tmp_vec *= (delta_t)*(1./4.);
    u_i[5] += tmp_vec;
    tmp_vec = u_i[0];
    u_i[5] += tmp_vec;
    
//    applyLimiters(u_i[5]);
    
        //! Final solution for the timestep
    *(this->current_solution) = u_i[5];
    
    cur_time = old_time + delta_t;
    
}

template <int dim>
void ExplicitTimeDiscretization<dim>::advance_SSP_5_4(void)
{
    n_timestep++;
    
    double old_time = cur_time;
    
    pcout << sc::nl << "Timestep: " << n_timestep << sc::nl
              << "Global time = " << cur_time << sc::nl;
    
//    applyLimiters(*(this->current_solution));
//    current_solution->update_ghost_values();
    
        //! Set the old_solution equation to the current
        //! The "current" solution is then the initial guess
        //! for the Newton iterations
    set_old_from_current();
    
        //! Create temp vector
    ParVec &tmp_vec = stages[5];
    
    u_i[0] = *(this->current_solution);
    u_i[0].update_ghost_values();
    
//    std::cout<<"\nu_i[0] = "<<u_i[0]<<std::endl;
//    std::cout<<"\nstages = "<<stages[0]<<std::endl;
//    getchar();
//    
//    std::cout<<"\nl2 norm of solution = "<<u_i[0].l2_norm();
    
        //!  Note:  stages[j] = RHS(u_i[j])
    
        //! Stage 1
    compute_residual(0,0);
    
    tmp_vec = stages[0];
    tmp_vec *= (delta_t)*(0.391752226571890);
    u_i[1] = tmp_vec;
    u_i[1] += u_i[0];
    
    u_i[1].update_ghost_values();
    applyLimiters(u_i[1], u_i[0]);
    
//    std::cout<<"\nu_i[1] = "<<u_i[1]<<std::endl;
//    std::cout<<"\nstages = "<<stages[1]<<std::endl;
//    getchar();
//    std::cout<<"\nl2 norm of solution = "<<u_i[1].l2_norm();
    
    cur_time = old_time + 0.39175222700392*delta_t;
    
        //! Stage 2
    compute_residual(1,1);
    tmp_vec = stages[1];
    tmp_vec *= (delta_t)*(0.368410593050371);
    u_i[2] = tmp_vec;
    tmp_vec = u_i[1];
    tmp_vec *= 0.555629506348765;
    u_i[2] += tmp_vec;
    tmp_vec = u_i[0];
    tmp_vec *= 0.444370493651235;
    u_i[2] += tmp_vec;
    
    u_i[2].update_ghost_values();
    applyLimiters(u_i[2], u_i[1]);
    
    
    cur_time = old_time + 0.58607968896780*delta_t;
    
        //! Stage 3
    compute_residual(2,2);
    tmp_vec = stages[2];
    tmp_vec *= (delta_t)*(0.251891774271694);
    u_i[3] = tmp_vec;
    tmp_vec = u_i[2];
    tmp_vec *= 0.379898148511597;
    u_i[3] += tmp_vec;
    tmp_vec = u_i[0];
    tmp_vec *= 0.620101851488403;
    u_i[3] += tmp_vec;
    
    u_i[3].update_ghost_values();
    applyLimiters(u_i[3], u_i[2]);
    
    cur_time = old_time + 0.47454236302687*delta_t;
    
        //! Stage 4
    compute_residual(3,3);
    tmp_vec = stages[3];
    tmp_vec *= (delta_t)*(0.544974750228521);
    u_i[4] = tmp_vec;
    tmp_vec = u_i[3];
    tmp_vec *= 0.821920045606868;
    u_i[4] += tmp_vec;
    tmp_vec = u_i[0];
    tmp_vec *= 0.178079954393132;
    u_i[4] += tmp_vec;
    
    u_i[4].update_ghost_values();
    applyLimiters(u_i[4], u_i[3]);
    
    cur_time = old_time + 0.93501063100924*delta_t;

        //! Stage 5
    compute_residual(4,4);
    tmp_vec = stages[4];
    tmp_vec *= (delta_t)*(0.226007483236906);
    u_i[5] = tmp_vec;
    tmp_vec = stages[3];
    tmp_vec *= (delta_t)*(0.063692468666290);
    u_i[5] += tmp_vec;
    tmp_vec = u_i[4];
    tmp_vec *= 0.386708617503269;
    u_i[5] += tmp_vec;
    tmp_vec = u_i[3];
    tmp_vec *= 0.096059710526147;
    u_i[5] += tmp_vec;
    tmp_vec = u_i[2];
    tmp_vec *= 0.517231671970585;
    u_i[5] += tmp_vec;

    u_i[5].update_ghost_values();
    applyLimiters(u_i[5], u_i[4]);
    
        //! Final solution for the timestep
    *(this->current_solution) = u_i[5];
    
        //! Note:  the code below counts ghost cells twice!  
    const double locally_relevant_norm = u_i[5].l2_norm();
    
    const double total_norm = std::sqrt(Utilities::MPI::sum (locally_relevant_norm, MPI_COMM_WORLD));
    
    pcout<<"\nl2 norm of solution = "<<total_norm<<std::endl;
    
    cur_time = old_time + delta_t;
    
}//advance_SSP_5_4

template class ExplicitTimeDiscretization<deal_II_dimension>;
