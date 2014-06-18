#ifndef DOMAIN_H
#define DOMAIN_H
#include <fstream>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/path_search.h>
#include <deal.II/base/conditional_ostream.h>

#include <fe/mapping.h>
#include <dofs/dof_handler.h>

using namespace dealii;

/** 
  Domain object. 
*/
template <int dim>
class Domain : public Subscriptor
{
  public:
      /** Empty constructor. */
      Domain (MPI_Comm& mpi_communicator);

      /** Full constructor. */
      Domain (MPI_Comm& mpi_communicator,
              ParameterHandler &prm);

      ~Domain ();

      /** Reinit. */
      void reinit(ParameterHandler &prm);

      /** Read mesh file name, etc. */
      void parse_parameters(ParameterHandler &prm);

      /** Generate entries in the given parameter file. */
      static void declare_parameters(ParameterHandler &prm);

      /** Generate the mesh. In this program, the mesh can be read from an
	input file generated with gmsh (http://www.geuz.org/gmsh/). */
      void create_mesh ();
      
      /** Write the mesh. */
//      void output_mesh(std::ostream &out) const;
      

      /** Write the mesh on the file specified by the parameter handler. */
//      void output_mesh() const;

      /** Reference to the triangulation. */
      inline parallel::distributed::Triangulation<dim> & get_tria() 
        {
        Assert(initialized, ExcNotInitialized());
        return tria;
        }//get_tria

  private:
    
      ConditionalOStream pcout;
    
      bool initialized;

      bool read_mesh;

      PathSearch search_mesh;

      std::string input_mesh_file_name;
      std::string input_mesh_format;
      std::string output_mesh_file_name;
    
      double ll_x, ll_y, ll_z;
      double ur_x, ur_y, ur_z;
      double n_cells_x, n_cells_y, n_cells_z;
      
      /** Holds the coarse triangulation. */
      parallel::distributed::Triangulation<dim>   tria;
      /** Helper class to output the grid. */
//      GridOut gridout;
};
#endif
