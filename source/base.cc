#include "../include/base.h"

template <int dim>
Base<dim>::Base () 
 :
 mpi_communicator (MPI_COMM_WORLD),                                                                                   
 domain (mpi_communicator),
 vspace (mpi_communicator),
 pcout (std::cout,
       (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{}

//we add a destructor
template <int dim>
Base<dim>::~Base ()
{}

template <int dim>
void Base<dim>::run (std::string &prm_file_name) 
{
    pcout << "Using Specified Parameter File: "
          << prm_file_name << std::endl;
    
  // ==============================
  declare_parameters ();
  
  prm.read_input(prm_file_name); 

  parse_parameters ();
   // ==============================
  
  /** Loop over the convergence cycles, top level loop */
  for (unsigned int cc=0; cc<number_of_cc; ++cc) 
  {  
    reinit_cc(cc);
    run_cc(cc); 
  }
  
//  error_handler.output_table();
  
}

template <int dim>
void Base<dim>::reinit_cc(unsigned int cc)
{
  /* redistribute the dofs and make hanging nodes
     insert information about what components go in what 
     blocks here */
  Timer timer;
  timer.start();
  pcout << "Refining and Redistributing DoFs..." << std::endl;
  /* if we are at more than one cc, refine the mesh */
  if(cc > 0) {

    Vector<float> error_estimates(vspace.get_tria().n_active_cells());
    typename FunctionMap<dim>::type neumann_boundary;
    //  KellyErrorEstimator<dim>::estimate (static_cast<DoFHandler<dim> &>(vspace.get_dh()),
    //					QGauss<dim-1>(std::ceil((2.0*double(vspace.get_fe().degree) + 1.0)/2.0)),
    //					neumann_boundary,
    //					data.u,
    //					error_estimates);

    vspace.flag_for_refinement(error_estimates);
    vspace.get_tria().execute_coarsening_and_refinement();

  }

  vspace.redistribute_dofs();
  timer.stop();
  pcout << "Refinement and DoFs Redistributed in: " 
  	  << timer.wall_time() << " seconds." << std::endl;

  pcout << "Triangulation Memory Consumption: " 
	  << vspace.get_tria().memory_consumption()/1024.0/1024.0
	  << " MB"
	  << std::endl;

  pcout << "DoFHandler Memory Consumption: " 
	  << vspace.get_dh().memory_consumption()/1024.0/1024.0
	  << " MB"
	  << std::endl;

  /* resize the error solution */
//  error_solution.reinit(vspace.n_dofs());

  /* reinit the other classes */
  advection.reinit(vspace);
}

template <int dim>
void Base<dim>::run_cc(unsigned int /*cc*/)
{
    advection.run(data_out_path);

  /* set the error solution to the solution from 
     the laplace class */
  //error_solution = data.u;

}

template <int dim>
void Base<dim>::declare_parameters()
{
  Domain<dim>::declare_parameters(prm);
  VectorSpace<dim>::declare_parameters(prm, "Vector Space Parameters");
  prm.enter_subsection("Vector Space Parameters");
  prm.set("Finite element space","FESystem[FE_Q(2)^"+Utilities::int_to_string(dim)+"]");
  prm.set("Dof ordering", "cuth");
  prm.leave_subsection();

  prm.enter_subsection("Data Parameters");
  prm.declare_entry("Output path",
		    "./out",
		    Patterns::Anything(),
		    "The path to write the data files to");
  prm.declare_entry("Write VTU",
		    "true",
		    Patterns::Bool(),
		    "Enable writing of VTU data files");
  prm.leave_subsection();

  /* For external classes */
  //elastic.declare_parameters(prm);
  //navier.declare_parameters(prm);
  advection.declare_parameters(prm);

}

template <int dim>
void Base<dim>::parse_parameters()
{
  Timer timer;
  timer.start();
    
  pcout << "Creating Domain and VectorSpace..." << std::endl;
  domain.reinit(prm);
  vspace.reinit(prm, domain.get_tria(), 1, 0, "Vector Space Parameters");
   timer.stop();
  pcout << "Domain and VectorSpace created in: " 
  	  << timer.wall_time() << " seconds." << std::endl;
    
    number_of_cc = 1;

  prm.enter_subsection("Data Parameters");
  data_out_path = prm.get("Output path");
  FSIutilities::createDirectory(data_out_path.c_str()); //.c_str() converts a string to a const char *
  write_vtu = prm.get_bool("Write VTU");
  prm.leave_subsection();
  pcout << "Writing data files to: "
	  << data_out_path 
	  << std::endl;

  /* For external classes */
 // elastic.parse_parameters(prm);
  advection.parse_parameters(prm);
  
  // material.parse_parameters(prm);
  // load.parse_parameters(prm);
  // elastic.parse_parameters(prm);

}

template class Base<deal_II_dimension>;
