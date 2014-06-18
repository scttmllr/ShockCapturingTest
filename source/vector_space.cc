#include "../include/vector_space.h"
#include <base/logstream.h>
#include <grid/grid_generator.h>
#include <grid/grid_in.h>
#include <grid/grid_out.h>
#include <grid/tria_iterator.h>
#include <grid/tria_accessor.h>
#include <fe/mapping_cartesian.h>
#include <fe/mapping_q.h>
#include <fe/fe.h>
#include <base/utilities.h>
#include <dofs/dof_renumbering.h>
#include <dofs/dof_tools.h>
#include <numerics/solution_transfer.h>
#include <numerics/vector_tools.h>
#include <grid/grid_tools.h>
#include <grid/grid_refinement.h>

using namespace std;

template <typename TYPE>
void smart_delete (SmartPointer<TYPE> &sp) {
  if(sp) {
    TYPE * p = sp;
    sp = 0;
    delete p;
  }
}

template <int dim> 
VectorSpace<dim>::VectorSpace(MPI_Comm& mpi_communicator) :
    pcout (std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    fe(0, "Vector Space FE"),
    dh(0, "Vector Space This DH"),
    tria(0, "Vector Space This Grid"),
    mapping(0, "Vetor Space Mapping"),
    mpi_communicator(mpi_communicator)
{}

template <int dim> 
VectorSpace<dim>::~VectorSpace()
{
  smart_delete(dh);
  smart_delete(fe);
  tria = 0;
  smart_delete(tria);
  smart_delete(mapping);
}


template <int dim>
void VectorSpace<dim>::refine_globally() 
{
    tria->refine_global (1);
    redistribute_dofs();
    deallog.pop();
}

template <int dim>
void VectorSpace<dim>::flag_for_refinement(Vector<float> &error) 
{
//    deallog.push("Refining");
    pcout << "Refining::Strategy: " << refinement_strategy << std::endl;
    if(refinement_strategy == "global") {
      tria->set_all_refine_flags();
    } else if(refinement_strategy == "fixed_number") {
      if(max_cells > 0)
	GridRefinement::refine_and_coarsen_fixed_number(*tria, error, 
							top_fraction, bottom_fraction, max_cells);
      else
	GridRefinement::refine_and_coarsen_fixed_number(*tria, error, 
							top_fraction, bottom_fraction);
    } else if(refinement_strategy == "fixed_fraction") {
      if(max_cells > 0)
	GridRefinement::refine_and_coarsen_fixed_fraction(*tria, error, 
							  top_fraction, bottom_fraction, max_cells);
      else
	GridRefinement::refine_and_coarsen_fixed_fraction(*tria, error, 
							  top_fraction, bottom_fraction);
    } else if(refinement_strategy == "optimize") {
      GridRefinement::refine_and_coarsen_optimize(*tria, error);
    } else {
      Assert(false, ExcInternalError());
    }
//    deallog.pop();
}



template <int dim>
void VectorSpace<dim>::reorder(int level,
			       std::vector<unsigned int> target_comps) {

    if(level<0) level = dh->get_tria().n_levels()-1;

    // Renumber by subdomain association only if needed
    if(n_mpi_processes > 1) {
        pcout << "Renumbering subdomain wise." << std::endl;
        DoFRenumbering::subdomain_wise(*dh);
    }
}//reorder

template <int dim>
void VectorSpace<dim>::measure_mesh() {
  typename Triangulation<dim>::active_cell_iterator cell, endc;
  endc = tria->end();
  h_max = 0;
  h_min = 1000;
  for(cell = tria->begin_active(); cell != endc; ++cell) {
    h_max = std::max(h_max, cell->diameter());
    h_min = std::min(h_min, cell->diameter());
  }
  pcout << "Max diameter of a cell: " << h_max << std::endl
	  << "Min diameter of a cell: " << h_min << std::endl;
}

template <int dim>
void VectorSpace<dim>::reinit(ParameterHandler &prm, 
			      parallel::distributed::Triangulation<dim> &dd,
			      const unsigned int n_mpi,
			      const unsigned int this_mpi,
			      const std::string space_name) 
{
//  deallog.push("VECTORSPACE");
  n_mpi_processes = n_mpi;
  current_mpi_process = this_mpi;
  
  smart_delete(dh);
  smart_delete(fe);
  smart_delete(mapping);

  // Parse Parameters
  parse_parameters(prm, space_name);

  // generate Finite element
  FiniteElement<dim> * fe_p=0;
  
  fe_p = FETools::get_fe_from_name<dim>(fe_name);
    
  fe = fe_p;
  pcout << "Finite Element Space: " << fe->get_name() << endl;

  if(map_type == 0) {
    mapping= new MappingCartesian<dim>();
  } else {
    mapping = new MappingQ<dim>(map_type);
  }
  
  tria = &dd;

  initialize_mesh();

  // Generate new DoFHandlers
  dh = new DoFHandler<dim>(*tria);
  
  // Now generates the boundary maps
  std::vector<std::vector<std::string> > bcs;
  bcs.push_back(d_bc);
  bcs.push_back(n_bc);
  bcs.push_back(o_bc);

  std::vector<std::map<char, std::vector<bool> > > bc_maps(3);
    
  for(unsigned int bcnumber=0; bcnumber<3; ++bcnumber) {
    // Get some aliases for the bcs
    std::vector<std::string> & bc = bcs[bcnumber];
    std::map<char, std::vector<bool> > & bc_map = bc_maps[bcnumber];
      
    if(bc.size()) 
      for(unsigned int i=0; i < bc.size(); ++i) {
	std::vector<std::string> id_and_comps = 
	  Utilities::split_string_list(bc[i], ':');
	AssertThrow(id_and_comps.size() == 2,
		    ExcMessage("Wrong Format for boundary indicator map."));
	std::vector<int> ids = 
	  Utilities::string_to_int(Utilities::split_string_list(id_and_comps[0]));
	std::vector<int> comps = 
	  Utilities::string_to_int(Utilities::split_string_list(id_and_comps[1]));
	  
	unsigned int n_c = fe->n_components();
	std::vector<bool> filter(n_c, false);
	// Now check that the components make sense
	for(unsigned int i=0; i<comps.size(); ++i) {
	  AssertThrow((unsigned int) comps[i] < n_c,
		      ExcIndexRange(comps[i], 0, n_c));
	  filter[ comps[i] ] = true;
	}
	// And now save these components. Merge them if the map is
	// already stored.
	for(unsigned int i = 0; i<ids.size(); ++i) {
	  // For each of the id, save the map just generated
	  if(bc_map.find(ids[i]) != bc_map.end()) {
	    // For each of the id, add the components that appear in
	    // the map just generated.
	    Assert(bc_map[ ids[i] ].size() == (unsigned int) n_c,
		   ExcDimensionMismatch(bc_map[ ids[i] ].size(), n_c));
	    for(unsigned int j=0; j<n_c; ++j) {
	      bc_map[ ids[i] ][j] = filter[j];
	    }
	  } else {
	    bc_map[ids[i]] = filter;
	  }
	}
      }
  }
  dirichlet_bc = bc_maps[0];
  neumann_bc = bc_maps[1];
  other_bc = bc_maps[2];

//  deallog.pop();
}

template <int dim>
void VectorSpace<dim>::initialize_mesh() 
{
  tria->refine_global (global_refinement);
  pcout << "Active Cells: "
	  << tria->n_active_cells()
	  << endl;
}

template <int dim>
void VectorSpace<dim>::redistribute_dofs(std::vector<unsigned int> target_components ) {
    
    // Measure it
    measure_mesh();
    
    // And Distribute degrees of freedom
    dh->distribute_dofs(*fe);
    
        //! Get the dof's owned by this processor
    locally_owned_dofs = dh->locally_owned_dofs ();
    
        //! Get the locally ownded dofs + ghost cell dofs
    locally_relevant_dofs.clear();
    
    DoFTools::extract_locally_relevant_dofs (*dh,
                                             locally_relevant_dofs);
}

template <int dim>
void VectorSpace<dim>::declare_parameters(ParameterHandler &prm, const std::string space_name) 
{

   prm.enter_subsection(space_name);

    prm.declare_entry ("Finite element space", "FE_Q(1)", 
		       Patterns::Anything(),
		       "The finite element space to use. For vector "
		       "finite elements use the notation "
		       "FESystem[FE_Q(2)^2-FE_DGP(1)] (e.g. Navier-Stokes). ");

    prm.declare_entry ("Mapping degree", "1", Patterns::Integer(),
		       "Degree of the mapping. If 0 is used, then a Cartesian mapping is assumed.");
    prm.declare_entry ("Dof ordering", "cuth, comp", Patterns::Anything(),
		       "Ordering of the degrees of freedom: none, comp, cuth, upwind.");
    prm.declare_entry ("Wind direction", ".01, .01, 1", Patterns::Anything(),
		       "Direction of the wind for upwind ordering of the mesh. ");

    prm.declare_entry ("Dirichlet boundary map", "1:0", Patterns::Anything(),
		       "Boundary indicator, followed by semicolomn and a list"
		       " of components to which this boundary conditions apply. "
		       "More boundary indicators can be separated by semicolumn. "
		       "1:0,1,4 ; 2,4:0,2");
    prm.declare_entry ("Neumann boundary map", "2:0", Patterns::Anything(),
		       "Boundary indicators, followed by semicolomn and a list of "
		       "components to which this boundary conditions apply. More "
		       "boundary indicators can be separated by semicolumn. "
		       "1:0,1,4 ; 2,4:0,2");
    prm.declare_entry ("Other boundary map", "3:0", Patterns::Anything(),
		       "Boundary indicator, followed by semicolomn and a list of "
		       "components to which this boundary conditions apply. More "
		       "boundary indicators can be separated by semicolumn. "
		       "1:0,1,4 ; 2,4:0,2");
    
    prm.enter_subsection("Grid Parameters");
    
    prm.declare_entry ("Global refinement", "4", Patterns::Integer());
    prm.declare_entry ("Distortion coefficient", "0", Patterns::Double(),
		       "If this number is greater than zero, the mesh is distorted"
		       " upon refinement in order to disrupt its structureness.");
    
    prm.declare_entry("Refinement strategy", 
		      "fixed_number", Patterns::Selection("fixed_number|fixed_fraction|optimize|global"),
		      "fixed_number: the Top/Bottom threshold fraction of cells are flagged for "
		      "refinement/coarsening. "
		      "fixed_fraction: the cells whose error is Top/Bottom fraction of the total "
		      "are refined/coarsened. optmized: try to reach optimal error distribution, "
		      "assuming error is divided by 4 upon refining. global: refine all cells.");
    prm.declare_entry("Bottom fraction", ".3", Patterns::Double());
    prm.declare_entry("Top fraction", ".3", Patterns::Double());
    prm.declare_entry("Max number of cells", "0", Patterns::Integer(),
		      "A number of zero means no limit. ");
    prm.leave_subsection();
  
    
    prm.leave_subsection();
}

    template <int dim>
void VectorSpace<dim>::parse_parameters(ParameterHandler &prm, const std::string space_name) 
{
    prm.enter_subsection(space_name);

    fe_name = prm.get ("Finite element space"); 
    map_type = prm.get_integer("Mapping degree");
    std::string all_ordering = prm.get ("Dof ordering");

    d_bc =  Utilities::split_string_list(prm.get ("Dirichlet boundary map"), ';');
    n_bc =  Utilities::split_string_list(prm.get ("Neumann boundary map"), ';');
    o_bc =  Utilities::split_string_list(prm.get ("Other boundary map"), ';');
    
    std::vector<std::string> wind_str = 
	Utilities::split_string_list(prm.get ("Wind direction") );
    for(unsigned int i=0; (i<wind_str.size()) && (i<dim); ++i) 
	    sscanf(wind_str[i].c_str(), "%lf", &wind[i]);
				     
    prm.enter_subsection("Grid Parameters");
    
    global_refinement		= prm.get_integer ("Global refinement");
    distortion			= prm.get_double("Distortion coefficient");
    
    refinement_strategy		= prm.get("Refinement strategy");
    enable_local_refinement	= !(refinement_strategy == "global");
    
    bottom_fraction		= prm.get_double("Bottom fraction");
    top_fraction		= prm.get_double("Top fraction");
    max_cells			= prm.get_integer("Max number of cells");
    prm.leave_subsection();

    prm.leave_subsection();

    ordering = Utilities::split_string_list(all_ordering);
}


template <int dim>
void VectorSpace<dim>::interpolate_dirichlet_bc(const Function<dim> & f,  
						std::map<unsigned int, double> & bvalues) 
{
//  deallog.push("DBC");
  std::map<char, std::vector<bool> >::iterator 
    dmap = dirichlet_bc.begin(),
    dmapend = dirichlet_bc.end();
  
//  unsigned int last_counted = 0;
  for(; dmap!=dmapend; ++dmap) {
    char id = dmap->first;
    std::vector<bool> &filter = dmap->second;
    
//        deallog << (int) id << " :";
    const unsigned int n_components = fe->n_components();
    for(unsigned int i=0; i < n_components; ++i) {
//      if(filter[i]) deallog << i << ", ";
    }
    VectorTools::interpolate_boundary_values(*mapping, *dh, 
					     // Dirichlet boundary only...
					     id, f, bvalues, filter);
//    deallog << " #: " << bvalues.size() - last_counted << std::endl;
//    last_counted = bvalues.size();
  }
//  deallog << "Total DBC dofs: " << bvalues.size() << std::endl;
//  deallog.pop();
}
template class VectorSpace<deal_II_dimension>;
