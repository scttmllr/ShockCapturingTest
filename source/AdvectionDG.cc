#include "../include/AdvectionDG.h"

template <int dim>
AdvectionDG<dim>::AdvectionDG ()
    :
    pcout (std::cout,
       Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
    td(
       std_cxx1x::bind(&AdvectionDG<dim>::assemble_rhs, 
                       this, std_cxx1x::_1, std_cxx1x::_2, std_cxx1x::_3),
       std_cxx1x::bind(&AdvectionDG<dim>::assemble_mass_matrix_and_multiply, 
                       this, std_cxx1x::_1, std_cxx1x::_2),
       std_cxx1x::bind(&AdvectionDG<dim>::check_convergence, 
                       this, std_cxx1x::_1, std_cxx1x::_2),
       std_cxx1x::bind(&AdvectionDG<dim>::init_newton_iteration, this, std_cxx1x::_1),
       std_cxx1x::bind(&AdvectionDG<dim>::applyLimiters, this, std_cxx1x::_1, std_cxx1x::_2)
       ),
    U(u_comp),
    initial_conditions(1),
    boundary_conditions(max_n_boundaries)
{}
   
template <int dim>
AdvectionDG<dim>::~AdvectionDG ()
{}


template <int dim>
double AdvectionDG<dim>::compute_u_2d(double time, Point<dim> &point)
{
    double x=point[0];
    double y=point[1];
    double r0 = 0.15;
    
    double x0, y0, r;
    
    //cone
    x0 = 0.5;
    y0 = 0.25;
    
    r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)) / r0;
    
    if(r<r0)
    {
        return (1-r);
    }
    
    //hump
    x0 = 0.25;
    y0 = 0.5;
    
    r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)) / r0;
    
    if(r<r0)
    {
        return 0.25*(1+std::cos(nc::pi*r));
    }
    
    //slotted cylinder
    x0 = 0.5;
    y0 = 0.75;
    
    r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0)) / r0;
    
    if(r<r0)
    {
        if(std::abs(x-x0)>= 0.025 || std::abs(y-y0)>=0.85)
            return 1;
    }
    
    return 0;
    
}// compute_u_2d

template <int dim>
void AdvectionDG<dim>::reinit (VectorSpace<dim> &given_vspace)
{
    vspace = &given_vspace;
    
    td.set_vector_space(given_vspace);
}

template <int dim>
bool AdvectionDG<dim>::check_convergence (unsigned int iteration, 
                   ParVec& residual)
{
    double norm = residual.l2_norm();
    
    pcout<<sc::nl<<"Residual l2 norm = "<<norm;
    
    if (iteration == 0)
    {
        if (norm < nc::epsilon)
            return true;
        else
            initial_residual_norm = norm;
    }
    else if ( (norm/initial_residual_norm) < nc::epsilon ||
             norm < nc::epsilon)
            return true;
    
        //! Should probably read these epsilons in from a file.
    
    return false;
}//check_convergence

template <int dim>
void AdvectionDG<dim>::setup_system ()
{
    update_cell_data(true);
}//setup_system

/***********************************************//*
@function update_limiter_data
@return   void
***************************************************/
template <int dim>
void AdvectionDG<dim>::update_limiter_data (ParVec& solution,
                                                ParVec& old_solution,
                                                bool update_geometry_info)
{
    unsigned int num_local_cells = 0;
    
    //    if(update_geometry_info)
    {
        // Set the local (per proc) user index
        unsigned int i=0;
        for (typename DoFHandler<dim>::active_cell_iterator cell = vspace->get_dh().begin_active();
             cell != vspace->get_dh().end(); ++cell)
        {
            if (cell->is_locally_owned())
            {
                cell->set_user_index(i);
                ++i;
            }
        }
        
        num_local_cells = i;
        
        cell_data.resize(num_local_cells);
        cell_limiter_data.resize(num_local_cells);
        
        // Need to zero the cell_limiter_data.  Hack!!
        for (unsigned int i=0; i<num_local_cells; ++i){
            cell_limiter_data[i] = 0.;
            cell_data[i] = 0.;
        }
    }
    
    MeshWorker::IntegrationInfoBox<dim> info_box;

    cur_time = td.current_time();

    const unsigned int n_gauss_points = std::ceil(((2*vspace->get_fe().degree) +1)/2);
    info_box.initialize_gauss_quadrature(n_gauss_points,
                                         n_gauss_points,
                                         n_gauss_points);

    info_box.initialize_update_flags();
    UpdateFlags update_flags = update_quadrature_points |
                                update_values;
    info_box.add_update_flags(update_flags, true, true, true, true);
    info_box.add_update_flags(update_normal_vectors, false, true, true, true);

    NamedData<ParVec* > solution_data;
    ParVec* sol_ptr = &solution;
    solution_data.add(sol_ptr, "solution");
    
    ParVec* old_ptr = &old_solution;
    solution_data.add(old_ptr, "old_solution");

    info_box.cell_selector.add("solution", true, false, false);
    info_box.boundary_selector.add("solution", true, false, false);
    info_box.face_selector.add("solution", true, false, false);
    
    info_box.cell_selector.add("old_solution", true, false, false);
    info_box.boundary_selector.add("old_solution", true, false, false);
    info_box.face_selector.add("old_solution", true, false, false);

    info_box.initialize(vspace->get_fe(), vspace->get_mapping(), solution_data);

    MeshWorker::DoFInfo<dim> dof_info(vspace->get_dh());
    
    ParVec pippo(vspace->locally_owned_dofs,
                 vspace->locally_relevant_dofs,
                 vspace->mpi_communicator);
    
    MeshWorker::Assembler::ResidualSimple<ParVec > assembler;
    NamedData<ParVec* > data;
    ParVec* rhs = &pippo;
    data.add(rhs, "Residual");
    assembler.initialize(data);
    
    MeshWorker::LoopControl loop_control;

    MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >(
        CellFilter (IteratorFilters::LocallyOwnedCell(),
                   vspace->get_dh().begin_active()),
        CellFilter (IteratorFilters::LocallyOwnedCell(),
                   vspace->get_dh().end()),
        dof_info,
        info_box,
//        NULL,
//        NULL,
//        NULL,
        std_cxx1x::bind(&AdvectionDG<dim>::compute_avg_min_max,
                    this, std_cxx1x::_1, std_cxx1x::_2),
        std_cxx1x::bind(&AdvectionDG<dim>::integrate_boundary_discontinuity_indicator,
                    this, std_cxx1x::_1, std_cxx1x::_2),
        std_cxx1x::bind(&AdvectionDG<dim>::integrate_face_discontinuity_indicator,
                    this, std_cxx1x::_1, std_cxx1x::_2, std_cxx1x::_3, std_cxx1x::_4),
        assembler, loop_control);
    
}//update_limiter_data


template <int dim>
bool AdvectionDG<dim>::is_troubled(const typename DoFHandler<dim>::active_cell_iterator cell)
{
    userCellData& data = cell_data[cell->user_index()];
    
//    if(data.g_k_i_pressure < 1.e-8)
//        return false;
    
    data.h_k = cell->diameter();
    data.volK = cell->measure();
    data.volK_34 = std::pow(data.volK, 0.75);
    
    data.g_k_i /= ( (data.h_k)*(data.volK_34) );
    
    double u_thresh = 0.05;
    
    if(std::abs(data.g_k_i) > u_thresh)
    {
        data.G_k_i = 1.0;
        return true;
    }
    else
        data.G_k_i = 0.0;
    
        // Test:  activate limiting everywhere
//        data.G_k_i = 1.0;
//    return true;
    
    return false;
    
    // Test:  activate limiting everywhere
    //        data.G_k_i = 1.0;
}

/***********************************************//*
@function update_cell_data
@return   void
@parameters (int)timestep:  the timestep number we're on

@purpose  Assign userCellData objects to the cell's
  user pointer, and compute the data
  needed for the stabilization operator
***************************************************/
template <int dim>
void AdvectionDG<dim>::update_cell_data (bool update_geometry_info)
{
    
//    if(update_geometry_info)
    {
            // Set the local (per proc) user index
    unsigned int i=0;
    for (typename DoFHandler<dim>::active_cell_iterator cell = vspace->get_dh().begin_active();
         cell != vspace->get_dh().end(); ++cell)
        {
        if (cell->is_locally_owned())
            {
            cell->set_user_index(i);
            ++i;
            }
        }
        
        cell_data.clear();
        cell_data.resize(i);
        
            // Need to zero the cell data.  Hack!!
        for (unsigned int i=0; i<cell_data.size(); ++i)
            cell_data[i] = 0.;
    }
    
}//update_cell_data

template <int dim>
void AdvectionDG<dim>::compute_avg_min_max (DoFInfo& dinfo,
                                                   CellInfo& info)
{
        // Get the cell_limiter_data object for this cell;
    limiter_data<dim>& cld = cell_limiter_data[dinfo.cell->user_index()];
    cld.set_default();
    
    const FEValuesBase<dim>& fe_v = info.fe_values();
    
    unsigned int n_q_points = fe_v.n_quadrature_points;
    
        // Current values -- for averages
    std::vector<double> &new_u    = info.values[0][0];
    
        // Old values -- for min/max
    std::vector<double> &old_u    = info.values[1][0];
    
    const unsigned int npts = new_u.size();
    Assert(npts==n_q_points, ExcDimensionMismatch(npts,n_q_points));
    
    double cell_size = 0.;
    
    for (unsigned int point=0; point<n_q_points; ++point)
    {
            // Averages based on "current" solution
            // to maintain conservativeness
        cld.u_avg += new_u[point] * fe_v.JxW(point);
        
        cell_size += fe_v.JxW(point);
        
            // Min/max alpha computed based on "current" solution
        cld.u_min = cld.u_min < new_u[point] ? cld.u_min : new_u[point];
        cld.u_max = cld.u_max > new_u[point] ? cld.u_max : new_u[point];
    }//quad-pt-loop #1
    
    cld.u_avg /= cell_size;
    
//    std::cout<<"\navg = " << cld.u_avg;
    
}//compute_avg_min_max

template <int dim>
void AdvectionDG<dim>::integrate_boundary_discontinuity_indicator (DoFInfo& dinfo,
                                                                  CellInfo& info)
{
//    userCellData& ucd = cell_data[dinfo.cell->user_index()];
//    limiter_data<dim>& cld = cell_limiter_data[dinfo.cell->user_index()];
//    
//    const FEValuesBase<dim>& fe_v = info.fe_values();
//        
//    unsigned int n_q_points= fe_v.n_quadrature_points;
//    
//    const unsigned int boundary_id = dinfo.face->boundary_indicator();
//    
//    unsigned int b_index = get_boundary_index(boundary_id);
//    
//    if(boundary_conditions[b_index].type[1] != 0//boundary_conditions[b_index]::BoundaryType::symmetry 
//       && boundary_conditions[b_index].type[1] != 3)//boundary_conditions[b_index]::BoundaryType::slip)
//    {
//    
//        // Get the boundary values through the FunctionParser
//    std::vector<Vector<double> > input_values(n_q_points, Vector<double>(n_comp));
//        // Boundary values at the quadrature points
//    boundary_conditions[b_index].values.vector_value_list(fe_v.get_quadrature_points(),
//                                                          input_values);
//    
//    std::vector<double> &u_values    = info.values[0][0];
//    
//    const unsigned int npts = vel[1].size();
//    Assert(npts==n_q_points, ExcDimensionMismatch(npts,n_q_points));
//    
//    std::vector<Vector<double> > prescribed_values(n_q_points, Vector<double>(n_comp));
//    
//    boundary_conditions[b_index].values.vector_value_list(fe_v.get_quadrature_points(), prescribed_values);
//    
//        // The boundary values that depend on the prescribed values
//        // or the interior (trace) values
//    double u_star;
//    
//    for (unsigned int q=0; q<n_q_points; ++q)
//    {
//            // Compute the "star" values based on the prescribed
//            // boundary type, interior traces, and the necessary
//            // externally defined prescribed values
//    
//            // Integrate jumps over the interface
//            // 1.  Conserved variables
//        ucd.g_k_i_rho1alpha += (a_star*rho1star - alpha_values[q]*rho1)
//                    *(a_star*rho1star - alpha_values[q]*rho1)
//                    *fe_v.JxW(q);
//        
//
//            // Min/max alpha computed based on "current" solution
//        cld.a_min = cld.a_min < a_star ? cld.a_min : a_star;
//        cld.a_max = cld.a_max > a_star ? cld.a_max : a_star;
//        
//
//    }//q
//        
//    }//not-symmetry or slip

}//integrate_boundary_discontinuity_indicator


template <int dim>
void AdvectionDG<dim>::integrate_face_discontinuity_indicator (DoFInfo& dinfo1,
                                                              DoFInfo& dinfo2,
                                                              CellInfo& info1,
                                                              CellInfo& info2)
{
    limiter_data<dim>& cld = cell_limiter_data[dinfo1.cell->user_index()];
        // Get reference to current cell's data
    userCellData& ucd1 = cell_data[dinfo1.cell->user_index()];
    
    const FEValuesBase<dim>& fe_v = info1.fe_values();
    
        //return;
    unsigned int n_q_points = fe_v.n_quadrature_points;
    
    std::vector<double> &t_u    = info1.values[0][0];
    std::vector<double> &n_u    = info2.values[0][0];
    
    double val=0.;
    
    for (unsigned int q=0; q<n_q_points; ++q)
    {
            // Integrate jumps over the interface
            // 1.  Conserved variables
        val = (t_u[q] - n_u[q])
                * (t_u[q] - n_u[q])
                *fe_v.JxW(q);
        ucd1.g_k_i += val;
        
            // Min/max alpha computed based on "current" solution
        cld.u_min = cld.u_min < t_u[q] ? cld.u_min : t_u[q];
        cld.u_max = cld.u_max > t_u[q] ? cld.u_max : t_u[q];
    }//q
    
//    std::cout<<"\n u_min = "<<cld.u_min;
//    std::cout<<"\n u_max = "<<cld.u_max;
//    std::cout<<"\n u_avg = "<<cld.u_avg;
//    std::cout<<"\n g_k_i = "<<ucd1.g_k_i;
//    getchar();
    

}//integrate_face_discontinuity_indicator



/***********************************************//*
    @function projectInitialConditions
    @return   void
                                                  
    @purpose  Project the initial conditions to the
              current DG solution
***************************************************/
template <int dim>
void AdvectionDG<dim>::projectInitialConditions ()
{
    const unsigned int n_threads = multithread_info.n_default_threads;
	Threads::ThreadGroup<> threads;
	
	typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
	std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
    thread_ranges = Threads::split_range<active_cell_iterator> (CellFilter (IteratorFilters::LocallyOwnedCell(),
                                                                            vspace->get_dh().begin_active()),
                                                                CellFilter (IteratorFilters::LocallyOwnedCell(),
                                                                            vspace->get_dh().end()),
                                                                n_threads);
    
    AdvectionDG<dim>::projectInitialConditions_interval(thread_ranges[0].first,
                                                       thread_ranges[n_threads-1].second);
    
    td.set_current_from_old();
    
}

template <int dim>
void AdvectionDG<dim>::projectInitialConditions_interval (const typename DoFHandler<dim>::active_cell_iterator &begin,
                                                         const typename DoFHandler<dim>::active_cell_iterator &endc)
{
    const UpdateFlags update_flags = update_values
                                    | update_quadrature_points
                                    | update_JxW_values;
    
    QGauss<dim>   quadrature(std::ceil(((3.0*vspace->get_fe().degree+1))/2));
//    QGauss<dim-1>   face_quadrature(std::ceil(((3.0*vspace->get_fe().degree) +1)/2));
    
    FEValues<dim> fe_v (vspace->get_mapping(), vspace->get_fe(), quadrature, update_flags);
//    FEFaceValues<dim> fe_v_face (vspace->get_mapping(), vspace->get_fe(), face_quadrature, update_flags);
    
    Mapping<dim,dim> &mapping = vspace->get_mapping();
    
    const unsigned int dofs_per_cell   = vspace->get_fe().dofs_per_cell;
    
        // local mass matrix
    FullMatrix<double> u_v_matrix (dofs_per_cell, dofs_per_cell),
                        inv_matrix (dofs_per_cell, dofs_per_cell);
    
    Vector<double>  cell_vector (dofs_per_cell),
                    local_solution (dofs_per_cell);
    
    std::vector<unsigned int> dofs (dofs_per_cell);    
    
    typename DoFHandler<dim>::active_cell_iterator cell;
    
    Point<dim> real_point;
    double ic;
    
    limiter_data<dim> lcd;
    
    std::vector<double> u_values(fe_v.n_quadrature_points);
    
//    std::vector<double> u_face(fe_v_face.n_quadrature_points);
    
    for (cell=begin; cell!=endc; ++cell) //loops over the cells
        if (cell->is_locally_owned())
    {
        u_v_matrix  = 0.;
        inv_matrix  = 0.;
        cell_vector = 0.;
        
        fe_v.reinit (cell);
        
        cell->get_dof_indices (dofs);
        
        for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
        {
            real_point = fe_v.quadrature_point(point);
            
            if(dim==1)
            {
                if(real_point[0] < 0.20)
                    ic = 1.0;
                else
                    ic = 0.0;
            }
            else if (dim==2)
            {
                ic = compute_u_2d(0.0, real_point);
            }
            
            // Project the initial conditions onto our broken sobolev space.
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            {
                    // RHS
                cell_vector(i) += ic * fe_v[U].value(i, point) * fe_v.JxW(point);
                
                    // local Mass matrix
                for(unsigned int j=0; j<fe_v.dofs_per_cell; j++)
                {
                    u_v_matrix(i,j) +=  fe_v[U].value(i, point)
                                        * fe_v[U].value(j, point)
                                        * fe_v.JxW(point);
                }//j
                
            }//i
            
        }//point
        
        inv_matrix.invert(u_v_matrix);
        local_solution = 0.;
        inv_matrix.vmult(local_solution, cell_vector, false);
        
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i) {   
            td.access_old_solution()[dofs[i]] = local_solution(i);
        }
        
    }//cell loop
    
}//projectInitialConditions


/***********************************************//*
@function applyLimiters
@return   void

@purpose  Apply limiters to the DG solution.  This
        consists of two parts:
        1.  Enforce alpha \in [0,1]
        2.  Limit u,p at shocks
***************************************************/
template <int dim>
void AdvectionDG<dim>::applyLimiters (ParVec &solution,
                                             ParVec &old_solution)
{
        // Piecewise constant is diffusive enough
        // to be stable.  Don't need to do anything else.
    if (vspace->get_fe().degree==0)
        return;
    
//    std::cout<<"\nReturnign without applying limiters...";
//    return;
    
        //  We probably want to store the user_indices, as setting the
        //  `user_index' of a cell deletes this data.
        //  See step-39
    std::vector<unsigned int> old_user_indices;
    vspace->get_tria().save_user_indices(old_user_indices);
    
        // Update our vector of information regarding
        // solution smoothness on cells.
    // We need the following information for limiting:
    // 1.  Element averages of {a,u,p} from `current' solution
    // 2.  Min/max values of {u,p} from old_solution.
    // 3.  Discontinuity indicator (single scalar, maybe bool?)
    //      computed with `current' solution.
    //
    // Data structure:  On each processor, we make a data container
    // that is indexed by the cell_index() [which we actually enforce ourselves].
    // The container will hold objects of type `limiter_data' which we have
    // defined to be the data above.
    //    
    // Here we update all of the data:
    update_limiter_data(solution, old_solution);
    
    // And here we actually apply the limiter:
    cell_laplacian_limiter(solution);
    
//    getchar();
    
        // Restore the old values:
    vspace->get_tria().load_user_indices(old_user_indices);
}


template <int dim>
void AdvectionDG<dim>::cell_laplacian_limiter (ParVec &solution)
{
    const UpdateFlags update_flags = update_values
                                    | update_gradients
                                    | update_quadrature_points
                                    | update_JxW_values;
    
    const UpdateFlags face_update_flags = update_values
                                    | update_quadrature_points
                                    | update_normal_vectors
                                    | update_JxW_values;
    
    
    
    QGauss<dim>   quadrature(vspace->get_fe().degree+1);
    QGauss<dim-1>   face_quadrature(vspace->get_fe().degree+1);
    
    FEValues<dim> fe_v (vspace->get_mapping(), vspace->get_fe(), quadrature, update_flags);
    FEFaceValues<dim> fe_v_face (vspace->get_mapping(), vspace->get_fe(), face_quadrature, face_update_flags);
    
    Mapping<dim,dim> &mapping = vspace->get_mapping();
    
    const unsigned int u_dofs_per_cell = vspace->get_fe().dofs_per_cell;
    const unsigned int q_dofs_per_cell = dim*u_dofs_per_cell;
        // We must add 1 for the Lagrange multiplier that enforces the cell average
    const unsigned int dofs_per_cell = u_dofs_per_cell + q_dofs_per_cell;// + 1;
    
    const unsigned int lagrange_pos = dofs_per_cell - 1;
    
    unsigned int n_q_points = fe_v.n_quadrature_points;
    
        // local mass matrix
    FullMatrix<double> hdg_matrix (dofs_per_cell, dofs_per_cell);
    
    Vector<double>  local_rhs (dofs_per_cell),
                    local_solution (dofs_per_cell);
    
        // Data to hold the auxiliary vector q:
    Tensor<1,dim> q_i, q_j;
    double div_q;
    unsigned int iq, jq;
    
    // Positivity preserving stuff:
    double max_a = 1.0;
    double min_a = 0.0;
    
        // Real dofs for the cell
    std::vector<unsigned int> dofs (u_dofs_per_cell);
    
    typename DoFHandler<dim>::active_cell_iterator
                cell = vspace->get_dh().begin_active(),
                endc = vspace->get_dh().end();
 
    for (; cell!=endc; ++cell) //loops over the cells
        if (cell->is_locally_owned())
        {
//        std::cout<<"\ncell->user_index() = "<<cell->user_index()<<std::endl;
        // Is stabilization needed?
        userCellData& ucd = cell_data[cell->user_index()];
        limiter_data<dim>& lcd = cell_limiter_data[cell->user_index()];
            
            min_a = lcd.u_avg;
            max_a = lcd.u_avg;
            
        bool is_troubled_cell = is_troubled(cell);

        if(is_troubled_cell){
            
            std::cout<<"\nLimiting on cell "<<'\t'<<cell->user_index()<<std::endl;
            fe_v.reinit (cell);
            cell->get_dof_indices (dofs);
            
            hdg_matrix = 0.0;
            local_rhs = 0.0;
            
                //! Assemble the interior terms.  These are all matrix entries.
                //! CODE ONLY WORKS FOR dim==1 CURRENTLY!
            for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
                for (unsigned int i=0; i<u_dofs_per_cell; ++i)
                {
                iq = i + u_dofs_per_cell;
                q_i[0] = fe_v[U].value(i,point);
                    for(unsigned int j=0; j<u_dofs_per_cell; ++j)
                    {
                            // Probably need 2 dim loops, and multiply each of the u_dofs_per_cell
                            // by dim1 and dim2
                        jq = j + u_dofs_per_cell;
                        
                        div_q = fe_v[U].gradient(i,point)[0];
                        q_j[0] = fe_v[U].value(j,point);
                        
                            // Convective term
//                        hdg_matrix(i,j) +=  fe_v[U].value(i,point)*fe_v[U].gradient(j,point)[0]*fe_v.JxW(point);
                        
                            // The rest of the terms
                        hdg_matrix(i,jq) += fe_v[U].gradient(i,point)*q_j*fe_v.JxW(point);
                        hdg_matrix(iq,jq) += q_i*q_j*fe_v.JxW(point);
                        hdg_matrix(iq,j) += div_q*fe_v[U].value(j,point)*fe_v.JxW(point);
                    }
                    
//                local_rhs(i) -= 1000000*lcd.u_avg*fe_v[U].value(i,point)*fe_v.JxW(point);
                }
            
                //! Assemble the face terms:
            const double tau = 1.;
            for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
                fe_v_face.reinit(cell, face_no);
                
                if(cell->at_boundary(face_no))
                {
                    std::cout<<"\nNeed to implement limiter at boundaries"<<std::endl;
                    getchar();
                }
                
                limiter_data<dim>& nbd = cell_limiter_data[cell->neighbor(face_no)->user_index()];
                
                min_a = min_a < nbd.u_avg ? min_a : nbd.u_avg;
                max_a = max_a > nbd.u_avg ? max_a : nbd.u_avg;
                
//                std::cout<<"\n nbd = "<<nbd.u_avg;
                
                for (unsigned int point=0; point<fe_v_face.n_quadrature_points; ++point)
                    for (unsigned int i=0; i<u_dofs_per_cell; ++i)
                    {
                        iq = i + u_dofs_per_cell;
                        q_i[0] = fe_v_face[U].value(i,point);
                        
                        local_rhs(i) += tau*fe_v_face[U].value(i,point)
                                        *0.5*(nbd.u_avg+lcd.u_avg)
                                        *fe_v_face.JxW(point);
                        
                        local_rhs(iq) += 0.5*(nbd.u_avg+lcd.u_avg) * q_i[0]
                                        *fe_v_face.normal_vector(point)[0]
                                        *fe_v_face.JxW(point);
                        
                        for(unsigned int j=0; j<u_dofs_per_cell; ++j)
                        {
                            jq = j + u_dofs_per_cell;
                            q_j[0] = fe_v_face[U].value(j,point);
                            
                            hdg_matrix(i,jq) -= fe_v_face[U].value(i,point)*q_j[0]*
                                        fe_v_face.normal_vector(point)[0]*fe_v_face.JxW(point);
                            hdg_matrix(i,j) += tau*fe_v_face[U].value(i,point)
                                                *fe_v_face[U].value(j,point)*fe_v_face.JxW(point);
                        }
                    }
                
            }
            
                // Assemble the lagrange multiplier terms
//            for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
//            {
//                for (unsigned int i=0; i<u_dofs_per_cell; ++i)
//                {
//                    hdg_matrix(i,lagrange_pos) += fe_v[U].value(i,point)*fe_v.JxW(point);
//                    hdg_matrix(lagrange_pos,i) += fe_v[U].value(i,point)*fe_v.JxW(point);
//                }
//            
//            local_rhs(lagrange_pos) += lcd.u_avg*fe_v.JxW(point);
//            }
            
//            hdg_matrix.print(std::cout);
            
            hdg_matrix.gauss_jordan();
            local_solution = 0.;
            hdg_matrix.vmult(local_solution, local_rhs, false);
            
                // And now we use the gradient solution to post-process the desired
                // conservative variable:

            
            for (unsigned int i=0; i<u_dofs_per_cell; ++i)
            {
                    // Now we put in the global solution vector:
                    // Since the dofs for our current cell are
                    // independent of all other cells, we do not
                    // require a mutex lock here.
                solution[dofs[i]] = local_solution(i);
            }
            
            std::vector<double> u_values(u_dofs_per_cell);
            fe_v[U].get_function_values(solution, u_values);
            double new_avg=0., old_avg=0.;
            
            double u_min = 1.e8;
            double u_max = -1.e8;
            
            for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
            {
                new_avg += u_values[point]*fe_v.JxW(point);
                old_avg += lcd.u_avg*fe_v.JxW(point);
                
                u_min = u_min < u_values[point] ? u_min : u_values[point];
                u_max = u_max > u_values[point] ? u_max : u_values[point];
            }
            
//            std::cout<<"\n After limiting:";
//            std::cout<<"\n  old average = "<<old_avg;
//            std::cout<<"\n  new average = "<<new_avg;
//            std::cout<<"\n  difference = "<<std::abs(old_avg-new_avg);
//            std::cout<<std::endl;
            
            // Shift&scale positivity preserving:
            
//            std::cout<<"\n max_a = "<<max_a<<"\nmin_a = "<<min_a<<std::endl;
            
//            max_a = 1.;
//            min_a = 0.;
            
            // Compute a_theta:
            double tmp = std::abs(u_max-lcd.u_avg);
            if (tmp>nc::epsilon)
                tmp = std::abs(max_a-lcd.u_avg)/tmp;
            else
                tmp = 1.0;
            
            double a_theta = tmp < 1.0 ? tmp : 1.0;
            
            tmp = std::abs(u_min-lcd.u_avg);
            if (tmp>nc::epsilon)
                tmp = std::abs(min_a-lcd.u_avg)/tmp;
            else
                tmp = 1.0;
            
            a_theta = tmp < a_theta ? tmp : a_theta;
            
            Vector<double> cell_vector(u_dofs_per_cell);
            FullMatrix<double> u_v_matrix(u_dofs_per_cell,u_dofs_per_cell);
            
            cell_vector = 0.;
            u_v_matrix = 0.;
            
            for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
            {
                double a_limited = a_theta*u_values[point]    + (1.0-a_theta)*lcd.u_avg;
                
                for (unsigned int i=0; i<u_dofs_per_cell; ++i)
                {
                    // RHS
                    cell_vector(i) += a_limited * fe_v[U].value(i, point) * fe_v.JxW(point);
                    
                    // local Mass matrix
                    for(unsigned int j=0; j<u_dofs_per_cell; j++)
                    {
                        u_v_matrix(i,j) +=  fe_v[U].value(i, point)
                                            * fe_v[U].value(j, point)
                                            * fe_v.JxW(point);
                    }//j
                    
                }//i
                
            }//point
            
            for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
                fe_v_face.reinit(cell, face_no);
                fe_v_face[U].get_function_values(solution, u_values);
                
                for (unsigned int point=0; point<fe_v_face.n_quadrature_points; ++point)
                {
                    u_min = u_min < u_values[point] ? u_min : u_values[point];
                    u_max = u_max > u_values[point] ? u_max : u_values[point];
                }
                
            }
            
            u_v_matrix.gauss_jordan();
            local_solution = 0.;
            u_v_matrix.vmult(local_solution, cell_vector, false);
            
            for (unsigned int i=0; i<u_dofs_per_cell; ++i)
            {
                // Now we put in the global solution vector:
                // Since the dofs for our current cell are
                // independent of all other cells, we do not
                // require a mutex lock here.
                solution[dofs[i]] = local_solution(i);
            }
            
            // Check again
            fe_v[U].get_function_values(solution, u_values);
            new_avg=0., old_avg=0.;
            for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
            {
                new_avg += u_values[point]*fe_v.JxW(point);
                old_avg += lcd.u_avg*fe_v.JxW(point);
                
                u_min = u_min < u_values[point] ? u_min : u_values[point];
                u_max = u_max > u_values[point] ? u_max : u_values[point];
            }
            
//            std::cout<<"\n After positivity preserving:";
//            std::cout<<"\n  old average = "<<old_avg;
//            std::cout<<"\n  new average = "<<new_avg;
//            std::cout<<"\n  difference = "<<std::abs(old_avg-new_avg);
//            std::cout<<std::endl;
            
   
            }//end-if-troubled-cell
  
 /**************
                // Check to see what max and min of alpha are!
            fe_v[U].get_function_values(solution, u_values);  
  
                //! Need to compute the average values over the element
            u_min = u_values[0];
            u_max = u_min;

            for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
            {
                u_min = u_min < u_values[point] ? u_min : u_values[point];
                u_max = u_max > u_values[point] ? u_max : u_values[point];
            }//quad-pt-loop
            
//            if (u_min < (0.0-nc::epsilon))
//                std::cout<<"\nMinimum u = "<< u_min <<std::endl;
//            else if (u_max > (1.0+nc::epsilon))
//                std::cout<<"\nMaximum u = "<<u_max <<std::endl;
  ****************/
        }//cell loop (if locally owned)
    
}//enforce_positivity_interval



/***********************************************//*
@function finalize_timestep
@return   void

@purpose  After time is advanced, do any clean-up necessary
***************************************************/
template <int dim>
void AdvectionDG<dim>::finalize_timestep ()
{}

template <int dim>
void AdvectionDG<dim>::timestep_reinit ()
{ }

template <int dim>
void AdvectionDG<dim>::assemble_mass_matrix_and_multiply (ParVec& cur_solution,
                                                           ParVec& residual)
{
    const unsigned int n_threads = multithread_info.n_default_threads;
	Threads::ThreadGroup<> threads;
	
	typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
	std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
    thread_ranges = Threads::split_range<active_cell_iterator> (CellFilter (IteratorFilters::LocallyOwnedCell(),
                                                                            vspace->get_dh().begin_active()),
                                                                CellFilter (IteratorFilters::LocallyOwnedCell(),
                                                                            vspace->get_dh().end()),
                                                                n_threads);
    
    AdvectionDG<dim>::integrate_mass(cur_solution, 
                                            residual,
                                            thread_ranges[0].first,
                                            thread_ranges[n_threads-1].second);
}//assemble_mass_matrix_and_old_vector

template <int dim>
void AdvectionDG<dim>::assemble_rhs (ParVec& solution,
                                    ParVec& residual,
                                    bool assemble_jacobian)
{
    this->assemble_jacobian = assemble_jacobian;
    
    cur_time = td.current_time();
    
    MeshWorker::IntegrationInfoBox<dim> info_box;
    
    const unsigned int n_gauss_points = std::ceil(((2.0*vspace->get_fe().degree) +1)/2);
    info_box.initialize_gauss_quadrature(n_gauss_points,
                                         n_gauss_points,
                                         n_gauss_points);
    
    info_box.initialize_update_flags();
    UpdateFlags update_flags = update_quadrature_points |
                               update_values;
    
    info_box.add_update_flags_all(update_flags);
    info_box.add_update_flags(update_normal_vectors, false, true, true, true);
    info_box.add_update_flags(update_gradients, true, false, false, false);
    
    NamedData<ParVec* > solution_data;

    ParVec* soln = &solution;  
    
    solution_data.add(soln, "solution");
    
    info_box.cell_selector.add("solution", true, true, false);//values, gradients, hessians
    info_box.boundary_selector.add("solution", true, false, false);
    info_box.face_selector.add("solution", true, false, false);
       
    info_box.initialize(vspace->get_fe(), vspace->get_mapping(), solution_data);
    
    MeshWorker::DoFInfo<dim> dof_info(vspace->get_dh());
    
    MeshWorker::LoopControl loop_control;
    
        //! Here's a hack to avoid assembling the Jacobian if it is not necessary
    if(assemble_jacobian)
    {
        MeshWorker::Assembler::SystemSimple<SparseMatrix<double>, ParVec > assembler;
        
        assembler.initialize(stiffness, residual);
        
        MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
        (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     vspace->get_dh().begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     vspace->get_dh().end()),
         dof_info, info_box,
         std_cxx1x::bind(&AdvectionDG<dim>::integrate_cell_term, 
                        this, std_cxx1x::_1, std_cxx1x::_2),
         std_cxx1x::bind(&AdvectionDG<dim>::integrate_boundary_term, 
                         this, std_cxx1x::_1, std_cxx1x::_2),
         std_cxx1x::bind(&AdvectionDG<dim>::integrate_face_term, 
                         this, std_cxx1x::_1, std_cxx1x::_2, std_cxx1x::_3, std_cxx1x::_4),
         assembler, loop_control);
    }
    else
    {
        MeshWorker::Assembler::ResidualSimple<ParVec > assembler;
        NamedData<ParVec* > data;
        ParVec* rhs = &residual;
        data.add(rhs, "Residual");
        assembler.initialize(data);
        
        MeshWorker::loop<dim, dim, MeshWorker::DoFInfo<dim>, MeshWorker::IntegrationInfoBox<dim> >
        (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     vspace->get_dh().begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     vspace->get_dh().end()),
         dof_info, info_box,
         std_cxx1x::bind(&AdvectionDG<dim>::integrate_cell_term, 
                         this, std_cxx1x::_1, std_cxx1x::_2),
         std_cxx1x::bind(&AdvectionDG<dim>::integrate_boundary_term,
                         this, std_cxx1x::_1, std_cxx1x::_2),
         std_cxx1x::bind(&AdvectionDG<dim>::integrate_face_term,
                         this, std_cxx1x::_1, std_cxx1x::_2, std_cxx1x::_3, std_cxx1x::_4),
         assembler, loop_control);
    }
    
}//assemble_system (meshworker)

template <int dim>
void AdvectionDG<dim>::integrate_mass(ParVec& solution,
                                     ParVec& residual,
                                     const typename DoFHandler<dim>::active_cell_iterator &begin,
                                     const typename DoFHandler<dim>::active_cell_iterator &endc)
{
    cur_time = td.current_time();
    
    const UpdateFlags update_flags = update_values
                                    | update_quadrature_points
                                    | update_JxW_values;
    
    QGauss<dim>   quadrature(std::ceil(((2.0*vspace->get_fe().degree) +1)/2));
    
    FEValues<dim> fe_v (vspace->get_mapping(), vspace->get_fe(), quadrature, update_flags);
    
    unsigned int n_q_points = fe_v.n_quadrature_points;
    
    Mapping<dim,dim> &mapping = vspace->get_mapping();
    
    const unsigned int dofs_per_cell   = vspace->get_fe().dofs_per_cell;
    
        // local mass matrix
    FullMatrix<double> jacobian (dofs_per_cell, dofs_per_cell);
    
    Vector<double> local_residual (dofs_per_cell);
    
    std::vector<unsigned int> dofs (dofs_per_cell);    
    
    typename DoFHandler<dim>::active_cell_iterator cell;
    
    for (cell=begin; cell!=endc; ++cell) //loops over the cells
        if (cell->is_locally_owned()){
    
        fe_v.reinit (cell);
        
        cell->get_dof_indices(dofs);
        
        jacobian = 0.;
            
//            std::cout<<"\n residual = "<< residual(dofs[0]);
        
        for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
            for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
                for(unsigned int j=0; j<fe_v.dofs_per_cell; j++)
                    jacobian(i,j) +=      fe_v[U].value(i, point)
                                        * fe_v[U].value(j, point)
                                        * fe_v.JxW(point);
            
//            std::cout<<"\n jacobian = "<<jacobian(0,0);
//            std::cout<<"\n solution = "<<solution(dofs[0]);
        
        jacobian.gauss_jordan();
            
        for(unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            local_residual(i) = residual(dofs[i]);
            
        for(unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
            for(unsigned int j=0; j<fe_v.dofs_per_cell; j++)
                residual(dofs[i]) += jacobian(i,j)*local_residual(j);
            
//        std::cout<<"\n solution = "<<solution(dofs[0])<<std::endl;
        
        }//cell-loop
    
}//integrate_mass

    
template <int dim>
void AdvectionDG<dim>::integrate_cell_term(DoFInfo& dinfo,
                                          CellInfo& info)
{
    const FEValuesBase<dim>& fe_v = info.fe_values();
//    FullMatrix<double>& ui_vi_matrix = dinfo.matrix(0).matrix;
    Vector<double>&     cell_vector  = dinfo.vector(0).block(0);
    
    unsigned int n_q_points = fe_v.n_quadrature_points;
    
    std::vector<double> &u_values    = info.values[0][0];
    
    const std::vector<Tensor<1,dim> > &grad_u = info.gradients[0][0];

    const unsigned int npts = u_values.size();
    Assert(npts==n_q_points, ExcDimensionMismatch(npts,n_q_points));
    
    for (unsigned int point=0; point<fe_v.n_quadrature_points; ++point)
    {
        Tensor<1,dim> v;
        
        if(dim==1)
            v[0] = 1.;
        else
            {
                // 0.5 - y
            v[0] = 0.5 - fe_v.quadrature_point(point)[1];
                // x - 0.5
            v[1] = fe_v.quadrature_point(point)[0] - 0.5;
            }
        
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        {
            cell_vector(i) += u_values[point]*(v * fe_v[U].gradient(i, point)) * fe_v.JxW(point);
            
            //Now assemble the Jacobian of the residual vector (w.r.t. the unknowns)
//        if(assemble_jacobian)
//        {
//            FullMatrix<double>& ui_vi_matrix = dinfo.matrix(0).matrix;
//            for(unsigned int j=0; j<fe_v.dofs_per_cell; j++)
//            {
//                ui_vi_matrix(i,j) -=
//            }//j
//        }//if-assemble_jacobian
        }//i
    }//point
    
//    std::cout<<"\nCell-interior-residual = "<<cell_vector.l2_norm()<<std::endl;
//    std::cout<<"\ncell_vector = "<<cell_vector<<std::endl;
    
}//assemble-cell-term
    

/******************************************************//**
 * @function assemble_boundary_term
 * @purpose Given the interior trace of the solution,
 *   some boundary indicator, and prescribed values,
 *   we compute the "star" terms; that is, the values
 *   that should be on the boundary.  We then compute
 *   the fluxes based off of these terms.
 **********************************************************/                                                         
template <int dim>
void AdvectionDG<dim>::integrate_boundary_term(DoFInfo& dinfo,
                                              CellInfo& info)
{
//    std::cout<<"\n0"<<std::endl;
    const FEValuesBase<dim>& fe_v = info.fe_values();
//    FullMatrix<double>& ui_vi_matrix = dinfo.matrix(0).matrix;
    Vector<double>&     cell_vector  = dinfo.vector(0).block(0);
        //    std::cout << "IN ASSEMBLE_BOUNDARY_TERM" << std::endl;
    unsigned int n_q_points= fe_v.n_quadrature_points;
    
    const unsigned int boundary_id = dinfo.face->boundary_indicator();
    
//    unsigned int b_index = get_boundary_index(boundary_id);
    
//    std::cout << "boundary_id = " << boundary_id << std::endl;
//    std::cout << "boundary_index = " << b_index << std::endl;
    
    std::vector<double> &u_values    = info.values[0][0];
    
    const unsigned int npts = u_values.size();
    Assert(npts==n_q_points, ExcDimensionMismatch(npts,n_q_points));
    
        // The boundary values that depend on the prescribed values
        // or the interior (trace) values
    double u_star;
    
    Flux f;
//    FluxJacobian fj, bfj;
    
    for (unsigned int q=0; q<n_q_points; ++q)
    {
            // Hardcode the Dirichlet boundary values:
            // Let's try all zero as a test.
        if(dim==1){
            if(boundary_id == 0)
                f.u=1.0;
            else
                f.u=0.;
        }
        else{
            f.u = 0.0;
        }
        
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        {
            cell_vector(i) -= f.u * fe_v.normal_vector(q)[0]
                                    * fe_v[U].value(i,q)
                                    * fe_v.JxW(q);
            
                // Jacobian assembly
//        if(assemble_jacobian)
//        {
//            FullMatrix<double>& ui_vi_matrix = dinfo.matrix(0).matrix;
//            for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
//            {
//                    // First up:  Phase 1 mass balance
//                ui_vi_matrix(i,j) += fj.dFa_dUa * Da_Di * fe_v[alpha].value(j,q)
//                                        * fe_v[alpha].value(i,q)
//                                        * fe_v.JxW(q);
//                
//            }//j
//        }//if-assemble_jacobian
        }//i
        
    }//q
        
}//assemble_boundary_term
    
    
template <int dim>
void AdvectionDG<dim>::integrate_face_term (DoFInfo& dinfo1,
                                           DoFInfo& dinfo2,
                                           CellInfo& info1,
                                           CellInfo& info2)
{
    const FEValuesBase<dim>& fe_v = info1.fe_values();
    const FEValuesBase<dim>& fe_v_neighbor = info2.fe_values();
    
    Vector<double>& cell_vector_i = dinfo1.vector(0).block(0);
    Vector<double>& cell_vector_e = dinfo2.vector(0).block(0);

    unsigned int n_q_points = fe_v.n_quadrature_points;
    
    std::vector<double> &t_u    = info1.values[0][0];
    std::vector<double> &n_u    = info2.values[0][0];
    
    Flux f;
//    FluxJacobian tfj, nfj;
    
    
//    std::cout<<"\n quad pt = "<<fe_v.quadrature_point(0)<<std::endl;
    
//    std::cout<<"\n normal = "<<fe_v.normal_vector(0)[0];
    
    for (unsigned int q=0; q<n_q_points; ++q)
    {
            // Compute the flux:  u*(v.n)
        if(dim==1)
            {
                    // Upwind the flux: with v=1
                if( (t_u[q]) > (n_u[q]) )
                {
                    f.u = t_u[q]* fe_v.normal_vector(q)[0];
                }
                else
                {
                    f.u = n_u[q* fe_v.normal_vector(q)[0]];
                }
                
                f.u = t_u[q]* fe_v.normal_vector(q)[0];
            }
        else if (dim==2)
        {
            Tensor<1,dim> v;
                // 0.5 - y
            v[0] = 0.5 - fe_v.quadrature_point(q)[1];
                // x - 0.5
            v[1] = fe_v.quadrature_point(q)[0] - 0.5;
            
            double v_dot_n = v * fe_v.normal_vector(q);
            
                // Upwind the flux:
            if(v_dot_n>0.)//(t_u[q]*v_dot_n)>(n_u[q]*v_dot_n))
            {
                f.u = t_u[q]*v_dot_n;
            }
            else
            {
                f.u = n_u[q]*v_dot_n;
            }
        }
        
            //! Jacobian calc:
        for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        {
            cell_vector_i(i) -= f.u
                                * fe_v[U].value(i,q)
                                * fe_v.JxW(q);
            
            cell_vector_e(i) += f.u
                                * fe_v_neighbor[U].value(i,q)
                                * fe_v.JxW(q);
            
                // Jacobian assembly
        if(assemble_jacobian)
        {
            std::cout<<"\nNEED TO CHECK TO FOR NBR LOCALLY OWNED FOR IMPLICIT CALCS!"<<std::endl;
            exit(1);
            FullMatrix<double>& ui_vi_matrix = dinfo1.matrix(0,false).matrix;
            FullMatrix<double>& ue_vi_matrix = dinfo2.matrix(0,true).matrix;
            FullMatrix<double>& ui_ve_matrix = dinfo1.matrix(0,true).matrix;
            FullMatrix<double>& ue_ve_matrix = dinfo2.matrix(0,false).matrix;
    
//            for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
//            {
//                ui_vi_matrix(i,j) += (tfj.dF_du * fe_v[U].value(j,q) )
//                                        * fe_v[U].value(i,q)
//                                        * fe_v.JxW(q);
//            }//j
        }//if-assemble_jacobian
        }//i
        
    }//q
    
    //std::cout << "leaving assemble_face_term" << std::endl;	
}//assemble_face_term
            
            
template <int dim>
void AdvectionDG<dim>::init_newton_iteration (int stage)
{
//    linear_mass = 0.0;
//    nonlinear_mass = 0.0;
//    stiffness = 0.0;
    
//    if(stage==0)
//        update_cell_data(false);
//    else
//        cell_data.clear();
}


template <int dim>
void AdvectionDG<dim>::refine_grid (int max_level)
{
        //! For now, use the stabilization flags as the refinement indicators
        //  This means that I only want to refine around the discontinuities
        //  Good enough to get all of the code mechanisms working though.
    typename DoFHandler<dim>::active_cell_iterator
                                cell = vspace->get_dh().begin_active(),
                                endc = vspace->get_dh().end();
    
        //unsigned int cell_counter=0;
    double Gki, gki;
    unsigned int cell_no=0;
    for (; cell!=endc; ++cell)
  	{
        if (cell->is_locally_owned())
        {
        cell->clear_coarsen_flag( );
        cell->clear_refine_flag( );
            
//        cell->set_refine_flag();
//
        const userCellData& data = cell_data[cell_no];
        
        Gki = data.G_k_i;
        gki = data.g_k_i;
            
        if(Gki > 0.50 && cell->level() < max_level)
            cell->set_refine_flag();
        else if (Gki < 0.50 && cell->level() > 0)
            cell->set_coarsen_flag();
            
//            if(cell_no==0)
//            {
//                cell->set_refine_flag();
//            }
        
            // Refine based on alpha
//        if ( data.g_k_i_alpha > a_thresh && cell->level() < max_level)
//    		cell->set_refine_flag();
//        else if ( (data.g_k_i_pressure > p_thresh || data.g_k_i_velocity > u_thresh)
//                 && cell->level() < max_level/2)
//            cell->set_refine_flag();// set based on pressure jump
//    	else if ( gki < 1.e-6 && cell->level() > 0)
//    		cell->set_coarsen_flag();
        
//        if ( Gki > 0.5 && cell->level() < max_level)
//            if (caseName == "Advecting Interface")
//                cell->set_refine_flag(RefinementCase<dim>::cut_axis(0));
//            else//quad/octree
//                cell->set_refine_flag();
//    	else if ( Gki < 0.5 && cell->level() > 0)
//    		cell->set_coarsen_flag();
            
        ++cell_no;
        
        }//locally_owned
	}//end-cell-loop

//	std::vector<const ParVec *> transfer_in;
//	std::vector<ParVec *> transfer_out;
//    
//    ParVec& old_solution = td.access_old_solution();
//    ParVec& current_solution = td.access_current_solution();
//	
//	transfer_in.push_back(&current_solution);
//    transfer_in.push_back(&old_solution);
//	
//    parallel::distributed::SolutionTransfer<dim,ParVec >
//                                    soltrans(vspace->get_dh());
//    
//    vspace->get_tria().prepare_coarsening_and_refinement();
//	soltrans.prepare_for_coarsening_and_refinement(transfer_in);
//	
//	vspace->get_tria().execute_coarsening_and_refinement ();
//    
//    vspace->redistribute_dofs();
//    
//    ParVec interpolated_solution(vspace->locally_owned_dofs,
//               vspace->locally_relevant_dofs,
//               vspace->mpi_communicator);
//    
//    ParVec old_interpolated_solution(vspace->locally_owned_dofs,
//                                 vspace->locally_relevant_dofs,
//                                 vspace->mpi_communicator);
//    
//    transfer_out.push_back(&interpolated_solution);
//    transfer_out.push_back(&old_interpolated_solution);
//    
//    soltrans.interpolate(transfer_out);
//    
//    current_solution.reinit(interpolated_solution);
//    current_solution = interpolated_solution;
//    
//    old_solution.reinit(old_interpolated_solution);
//    old_solution = old_interpolated_solution;
//    
//    BoundaryIndicators<dim>::set_tria_flags(vspace, caseName);
//    
//    setup_system();
//        
//    td.reinit(vspace->locally_relevant_dofs.size(), td.current_time() );

}//refine_grid


template <int dim>
void AdvectionDG<dim>::output_results (int timestep, std::string data_out_path)
{ 
    DataOut<dim> data_out;
    data_out.attach_dof_handler (vspace->get_dh());
    
    Vector<float> subdomain (vspace->get_tria().n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
        subdomain(i) = vspace->get_tria().locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");
    
//    Vector<double> limiting (vspace->get_tria().n_active_cells());
//    typename DoFHandler<dim>::active_cell_iterator
//                    cell = vspace->get_dh().begin_active(),
//                    endc = vspace->get_dh().end();
//    unsigned int ii=0;
//    for (; cell!=endc; ++cell){
//        if(cell->is_locally_owned()){
//            limiting(ii) = cell_data[ii].g_k_i;
//            ++ii;
//        }
//    }
//    data_out.add_data_vector (limiting, "indicator");
    
    std::vector<std::string> solution_names(1, "u");
    
        // Data interpretation
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation (1, DataComponentInterpretation::component_is_scalar);
    
    data_out.add_data_vector (td.access_current_solution(), 
                              solution_names, 
                              DataOut<dim,DoFHandler<dim> >::type_automatic, 
                              interpretation);
        //data_out.add_data_vector (material_ids, "Material_ID");

    data_out.build_patches (vspace->get_fe().degree);
    
    const std::string filename = data_out_path +
            "/solution-" + Utilities::int_to_string (timestep, 6) +
            "." + Utilities::int_to_string (Utilities::MPI::
                                    this_mpi_process(MPI_COMM_WORLD),4);
    
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);
    
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD); ++i){
            filenames.push_back ("solution-" +
                                 Utilities::int_to_string (timestep, 6) +
                                 "." +
                                 Utilities::int_to_string (i, 4) +
                                 ".vtu");
         times_and_names.push_back (std::pair<double,std::string> (td.current_time(), filenames[i]));
        }
        
        std::ofstream master_output ((filename + ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
        
//        std::string tmp_name = filename + ".pvtu";
//        double tmp_time = td.current_time();
//        times_and_names.push_back (std::pair<double,std::string> (td.current_time(), filename + ".pvtu"));
        std::ofstream pvd_output ( (data_out_path + "/solution.pvd").c_str() );
        data_out.write_pvd_record (pvd_output, times_and_names);
    }
}//output_results


template <int dim>
void AdvectionDG<dim>::declare_parameters (ParameterHandler &prm)
{
    td.declare_parameters(prm);
    
    prm.enter_subsection ("AdvectionDG");
    {
        prm.declare_entry("Case name", "HSSV 2D", Patterns::Anything(), "Name for TwoPhase DG case");
        
        prm.declare_entry("Restart", "false", Patterns::Bool(), "Restart by reading file?");
        
        prm.enter_subsection("Adaptivity");
        {
            prm.declare_entry("IC refinements", "0", Patterns::Integer(), "IC refinements := adaptive refinement of ICs");
            prm.declare_entry("Levels", "0", Patterns::Integer(), "Levels := number of refinement levels");
            prm.declare_entry("Refinement frequency", "0", Patterns::Integer(), 
                              "Frequency := refine once every this many timesteps");
            
        }//Adaptivity
        prm.leave_subsection ();
        
        prm.enter_subsection("Output");
        {
            prm.declare_entry("Output frequency", "1", Patterns::Integer(), 
                              "Frequency := output results every this many timesteps");
            
        }//Output
        prm.leave_subsection ();

    }
    prm.leave_subsection ();
}

template <int dim>
void AdvectionDG<dim>::parse_parameters (ParameterHandler &prm)
{
    td.parse_parameters(prm);
    
    prm.enter_subsection("AdvectionDG");
    {
        caseName = prm.get("Case name");
        
        restart = prm.get_bool("Restart");
        
        prm.enter_subsection("Adaptivity");
        {
            ic_refinements = prm.get_integer("IC refinements");
            refine_levels = prm.get_integer("Levels");
            refine_freq = prm.get_integer("Refinement frequency");            
        }//Adaptivity
        prm.leave_subsection ();
        
        prm.enter_subsection("Output");
        {
            output_freq = prm.get_integer("Output frequency");
            
        }//Output
        prm.leave_subsection ();
    }
    prm.leave_subsection();
    
}//parse_parameters

template <int dim>
inline unsigned int AdvectionDG<dim>::get_boundary_index (unsigned int boundary_id) const
{
    for (unsigned int index=0; index<max_n_boundaries; ++index)
        if (boundary_conditions[index].boundary_id == boundary_id)
            return index;
    
    std::cout<<"\n  No matching boundary index for boundary_id = "<<boundary_id<<std::endl;
    return 0;
}//get_boundary_index

/***********************************************//**
  @function writeSolutionToDisk
  @arguments none
  @return void
  
  @purpose:  For non-adaptive cases, the solution can be used as a restart file.
  Or, it can be used as quasi-converged initial conditions
  *************************************************/
template<int dim>
void AdvectionDG<dim>::writeSolutionToDisk (int timestep, std::string data_out_path) const
{
        // Set file name for output
//    std::string fileName = data_out_path + "/solnVector-";
//    fileName += Utilities::int_to_string (timestep, 6);
//    fileName += ".dat";
//    
//    std::fstream fp;
//    fp.open(fileName.c_str(), std::ios::out);
//    
//    if(!fp.is_open() )
//    {
//        std::cout << "\nCannot open file " << fileName << std::endl;
//        exit(1);
//    }
//    
//    const Vector<double>& solution = td.access_current_solution();
//    
//    fp.precision(16);
//        // First thing we write is the length of the solution vector:
//    fp << solution.size() << std::endl;
//    for (unsigned int i=0; i<solution.size(); ++i)
//        fp << solution(i) << std::endl;
//    
//    
//        //fp << setprecision(16) << solution(i) << endl;
//    
//    fp.close();
    
}//writeSolutionToDisk

/**
 * The opposite of writing to the disk, we read back in
 * the data and put it into the solution vector
 */
template<int dim>
void AdvectionDG<dim>::readSolutionFromDisk (std::string data_out_path) 
{
        // Set file name for output
//    std::string fileName = data_out_path + "/solnVector.dat";
//    
//    std::fstream fp;
//    fp.open(fileName.c_str(), std::ios::in);
//    
//    if(!fp.is_open() )
//    {
//        std::cout << "\nCannot open file " << fileName << std::endl;
//        exit(1);
//    }
//    
//    fp.precision(16);
//    
//    Vector<double>& solution = td.access_current_solution();
//    
//        // Read the size:
//    double ds;
//    fp >> ds;
//    unsigned int size = static_cast<unsigned int>(ds);
//    
//        // Assert that the vector is sized appropriately:
//    if (solution.size() != size)
//    {
//        std::cout<<"\nSize of data = "<<size;
//        std::cout<<"\nSize of solution vector = "<<solution.size();
//        std::cout<<"\nExiting...\n"<<std::endl;
//        exit(1);
//    }
//    
//        // Need to handle the endl?
//    for (unsigned int i=0; i<size; ++i)
//        fp >> solution(i);
//        //fp << setprecision(16) << solution(i) << endl;
//    
//    fp.close();
    
}//writeSolutionToDisk


template <int dim>
void AdvectionDG<dim>::run (std::string data_out_path)
{
    setup_system ();
    
    unsigned int timestep=0;
    
    td.reinit(vspace->n_dofs());
//    td.set_mass_and_stiffness_matrices(&nonlinear_mass, &stiffness);
    
    cur_time = td.current_time();
    
        //    stab_factor *= 0.50*(vspace->h_min)*(td.time_step_size());
    
//    set_cell_user_pointer ();
    
//    BoundaryIndicators<dim>::set_tria_flags(vspace, caseName);

    if (restart)
    {
        readSolutionFromDisk(data_out_path);
    }
    else
        projectInitialConditions();
    
    td.update_ghost_cells();
    applyLimiters(td.access_current_solution(), td.access_old_solution());
    
//    pcout<<"\nAfter InitialConditions"<<std::endl;
    
//    update_cell_data (true);
    
//    output_results(0, data_out_path);
    if(!restart)
    for(int cycle=0; cycle<ic_refinements; ++cycle)
    {
        pcout<<"\nRefining initial conditions:  "<<cycle<<std::endl;
        
        refine_grid (ic_refinements);
        
        projectInitialConditions();
        td.update_ghost_cells();
//        applyLimiters(td.access_current_solution(), td.access_old_solution());
        
//        update_cell_data (true);
        
//        output_results(cycle, data_out_path);
    }
    
    td.update_ghost_cells();
 
//    exit(1);

    output_results (timestep, data_out_path);
    
    while (!td.finalized() )
    {
//        if(timestep>0)
//            readSolutionFromDisk();
        
//        if(refine_freq>0)
//            if(td.time_step_num() % refine_freq == 0)
//            {
//                refine_grid (refine_levels);
//                update_cell_data (true);
//            }

        td.advance();
        
        finalize_timestep();
        
        
//        std::cout<<"\nN_DOFS = "<<vspace->get_dh().n_dofs();
        
//        std::cout<<"\nl2 norm of solution = "<<(td.access_current_solution()).l2_norm();
        
            //        update_cell_data (false);
        
        if(refine_freq>0)
            if(td.time_step_num() % refine_freq == 0)
            {
                refine_grid (refine_levels);
                
                td.update_ghost_cells();
                applyLimiters(td.access_current_solution(), td.access_old_solution());
                td.update_ghost_cells();
                    //                update_cell_data (true);
            }
        
        if(output_freq>0)
        if (td.time_step_num() % output_freq == 0)
        {
            output_results (td.time_step_num(), data_out_path);
            
//            writeSolutionToDisk(td.time_step_num(), data_out_path);    
        }
        
//        if(refine_freq>0)
//            if(td.time_step_num() % refine_freq == 0)
//            {
//                refine_grid (refine_levels);
//                
//                if (caseName == "Notional Cav Body"){
//                    td.delta_t *= 0.5;
//                }
//                
//                td.update_ghost_cells();
//                applyLimiters(td.access_current_solution(), td.access_old_solution());
//                 td.update_ghost_cells();
////                update_cell_data (true);
//            }
        
        
    }
    
}//AdvectionDG::run()

template class AdvectionDG<deal_II_dimension>;
