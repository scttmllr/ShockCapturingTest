#include "../include/domain.h"
#include <base/logstream.h>
#include <base/utilities.h>
#include <grid/grid_generator.h>
#include <grid/grid_in.h>
#include <grid/tria_iterator.h>
#include <grid/tria_accessor.h>
#include <grid/grid_tools.h>
#include <numerics/vector_tools.h>
#include <fe/fe.h>

using namespace std;

template <int dim> 
Domain<dim>::Domain(MPI_Comm& mpi_communicator) :
    search_mesh("MESH", 1),
    pcout (std::cout,
             (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    tria (mpi_communicator)
//          typename Triangulation<dim>::MeshSmoothing
//                       (Triangulation<dim>::smoothing_on_refinement |
//                        Triangulation<dim>::smoothing_on_coarsening))
{
    initialized = false;
}

template <int dim>
Domain<dim>::Domain(MPI_Comm& mpi_communicator,
                    ParameterHandler &prm) :
    search_mesh("MESH", 1),
    pcout (std::cout,
             (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    tria (mpi_communicator)
//          typename Triangulation<dim>::MeshSmoothing
//                       (Triangulation<dim>::smoothing_on_refinement |
//                        Triangulation<dim>::smoothing_on_coarsening))
{
    reinit(prm);
}

template <int dim> 
Domain<dim>::~Domain()
{
  //     tria.clear();
}


template <int dim>
void Domain<dim>::reinit(ParameterHandler &prm) 
{
    parse_parameters(prm);
    pcout << "Generating coarse triangulation." << std::endl;
    create_mesh();
//    output_mesh();
    deallog.pop();
}


template <int dim>
void Domain<dim>::declare_parameters(ParameterHandler &prm) 
{
    prm.enter_subsection("Domain Parameters");
    prm.declare_entry ("Read domain mesh from file", "false", Patterns::Bool(),
		       "If this is false, then the input mesh file below is ignored and a hyper-cube is created.");
    prm.declare_entry ("Path of domain mesh files", "mesh/", Patterns::Anything());
    prm.declare_entry ("Input mesh file", "square", Patterns::Anything()); 
    prm.declare_entry ("Input mesh format", "ucd", 
		       Patterns::Selection(GridIn<dim>::get_format_names())); 
    prm.declare_entry ("Output mesh file", "square_out", Patterns::Anything()); 
    
    prm.declare_entry("Lower Left X", "0.0", Patterns::Double(),
                      "X coordinate of lower left corner");
    prm.declare_entry("Lower Left Y", "0.0", Patterns::Double(),
                      "Y coordinate of lower left corner");
    prm.declare_entry("Lower Left Z", "0.0", Patterns::Double(),
                      "Z coordinate of lower left corner");
    
    prm.declare_entry("Upper Right X", "0.0", Patterns::Double(),
                      "X coordinate of upper right corner");
    prm.declare_entry("Upper Right Y", "0.0", Patterns::Double(),
                      "Y coordinate of upper right corner");
    prm.declare_entry("Upper Right Z", "0.0", Patterns::Double(),
                      "Z coordinate of upper right corner");
    
    prm.declare_entry("Cells In X", "1", Patterns::Integer(),
                      "Number of cells in X direction");
    prm.declare_entry("Cells In Y", "1", Patterns::Integer(),
                      "Number of cells in Y direction");
    prm.declare_entry("Cells In Z", "1", Patterns::Integer(),
                      "Number of cells in Z direction");

//    prm.enter_subsection("Grid Out Parameters");
//    GridOut::declare_parameters(prm);
//    prm.leave_subsection();

    prm.leave_subsection();

}

template <int dim>
void Domain<dim>::parse_parameters(ParameterHandler &prm) 
{
    prm.enter_subsection("Domain Parameters");
    read_mesh		= prm.get_bool ("Read domain mesh from file");
    search_mesh.add_path(prm.get ("Path of domain mesh files"));
    input_mesh_file_name = prm.get ("Input mesh file");
    output_mesh_file_name = prm.get ("Output mesh file");
    input_mesh_format	= prm.get ("Input mesh format");

    if(!read_mesh)
    {
        ll_x = prm.get_double("Lower Left X");
        ll_y = prm.get_double("Lower Left Y");
        ll_z = prm.get_double("Lower Left Z");
        
        ur_x = prm.get_double("Upper Right X");
        ur_y = prm.get_double("Upper Right Y");
        ur_z = prm.get_double("Upper Right Z");
        
        n_cells_x = prm.get_integer("Cells In X");
        n_cells_y = prm.get_integer("Cells In Y");
        n_cells_z = prm.get_integer("Cells In Z");
    }
    
//    prm.enter_subsection("Grid Out Parameters");
//    gridout.parse_parameters(prm);
//    prm.leave_subsection();

    prm.leave_subsection();
}

template <int dim>
void Domain<dim>::create_mesh() 
{
  if(read_mesh) 
    {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation (tria);
    string mfilen = search_mesh.find
      (input_mesh_file_name, 
       grid_in.default_suffix(grid_in.parse_format(input_mesh_format)), 
       "r");
    ifstream mfile(mfilen.c_str());
    grid_in.read(mfile, GridIn<dim>::parse_format(input_mesh_format));
    } 
  else 
    {
    if (dim == 1)
        {
            const Point<dim> LowerLeft (ll_x),
            UpperRight (ur_x);
            
                // Define the subdivisions in the x1 and x2 coordinates.
            std::vector<unsigned int> subdivisions(dim);
            subdivisions[0] =   n_cells_x;
            
            GridGenerator::subdivided_hyper_rectangle(tria,
                                                      subdivisions,
                                                      LowerLeft,
                                                      UpperRight,
                                                      true);
        }
    else if (dim == 2)
        {
        const Point<dim> LowerLeft (ll_x, ll_y),
                        UpperRight (ur_x, ur_y );
        
            // Define the subdivisions in the x1 and x2 coordinates.
        std::vector<unsigned int> subdivisions(dim);
        subdivisions[0] =   n_cells_x;
        subdivisions[1] =   n_cells_y;
        
        GridGenerator::subdivided_hyper_rectangle(tria,
                                                  subdivisions,
                                                  LowerLeft,
                                                  UpperRight,
                                                  false);
        }
      else if (dim == 3)
        {
            const Point<dim> LowerLeft (ll_x, ll_y, ll_z),
            UpperRight (ur_x, ur_y, ur_z);
            
                // Define the subdivisions in the x1 and x2 coordinates.
            std::vector<unsigned int> subdivisions(dim);
            subdivisions[0] =   n_cells_x;
            subdivisions[1] =   n_cells_y;
            subdivisions[2] =   n_cells_z;
            
            GridGenerator::subdivided_hyper_rectangle(tria,
                                                      subdivisions,
                                                      LowerLeft,
                                                      UpperRight,
                                                      false);
        }

    }
  
  initialized = true;
}
  

template class Domain<deal_II_dimension>;
