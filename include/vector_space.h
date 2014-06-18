#ifndef VECTOR_SPACE_H
#define VECTOR_SPACE_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/persistent_tria.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/solution_transfer.h>

using namespace dealii;

/** 
  VectorSpace object. Anything related to the definition of the
  Hilbert space that makes up the pde problem lives in this class.
*/
template <int dim>
class VectorSpace : public Subscriptor
{
  public:
  
  /** Empty constructor. */
  VectorSpace (MPI_Comm& mpi_communicator);
  
  /** Destructor. Clears the created pointers. */
  ~VectorSpace ();
  
  /** Reinit. The state of this object after calling reinit, is
      the same as after the full constructor is used.*/
  void reinit(ParameterHandler &prm, 
	      parallel::distributed::Triangulation<dim> &dd,
	      const unsigned int n_mpi_processes=1,
	      const unsigned int current_mpi_process=0,
	      const std::string space_name="Vector Space Parameters");
  
  /** Parse the parameter file. */
  void parse_parameters(ParameterHandler &prm, const std::string space_name);

  /** Generate entries in the given parameter file. */
  static void declare_parameters(ParameterHandler &prm, const std::string space_name="Vector Space Parameters");

  /** Initialize the mesh. */
  void initialize_mesh();

  /** Refine the mesh globally. */
  void refine_globally();
  
  /** Flag the mesh according to the given error estimator and to the
      refinement strategy defined in the paratmeters. */
  void flag_for_refinement(Vector<float> &error);

  /** Transfer solution from old to new grid. */
  template <typename VECTOR>
  void transfer(VECTOR &dst, const VECTOR &src);

  /** Reorder the degrees of freedom according to the
      parameters. This is done for the level specified on the
      argument. If level is -1, then the active dofs are reordered. */
  void reorder(int level = -1,
	       std::vector<unsigned int> target_comps=std::vector<unsigned int>());

  /** Redistribute degrees of freedom. The optional arguments groups
      together the components. It is useful for vector valued finite elements, when one wants to sort together*/
  void redistribute_dofs(std::vector<unsigned int> 
			 target_components=std::vector<unsigned int>());
      
  /** Compute maximum and minimum diameter of the mesh cells. */
  void measure_mesh(); 
      
  /** Interpolate Boundary Conditions. Generates the boundary
      conditions for the problem. This will be used after the assembly
      procedures. */
  void interpolate_dirichlet_bc(const Function<dim> & f,  
				std::map<unsigned int, double> & bvalues);
   
  /** Return reference to finite element.*/
  inline FiniteElement<dim,dim> & get_fe() {
    return *fe;
  }

  /** Return reference to current dof handler.*/
    inline DoFHandler<dim,dim> & get_dh() {
    return *dh;
  }

  /** Return reference to current triangulation..*/
  inline parallel::distributed::Triangulation<dim,dim> & get_tria() {
    return *tria;
  }
      
  /** Return reference to mapping.*/
  inline Mapping<dim> & get_mapping() {
    return *mapping;
  }

      
  /** Return constant reference to finite element.*/
  inline const FiniteElement<dim,dim> & get_fe() const {
    return *fe;
  }

  /** Return constant reference to current dof handler.*/
    inline const DoFHandler<dim,dim> & get_dh() const {
    return *dh;
  }

  /** Return reference to current triangulation..*/
  inline const parallel::distributed::Triangulation<dim,dim> & get_tria() const {
    return *tria;
  }
      
  /** Return reference to mapping.*/
  inline const Mapping<dim,dim> & get_mapping() const {
    return *mapping;
  }
  
  /** Number of dofs. */
  inline unsigned int n_dofs() {
    return dh->n_dofs();
  }

  /** Number of dofs in this process. */
  inline  unsigned int n_ldofs() const {
    return n_local_dofs;
  }

  /** Number of blocks. */
  inline  unsigned int n_blocks() const {
    return number_of_blocks;
  }
  /** Bool that checks if local refiment is enabled.*/
  bool enable_local_refinement;
  
  /** Initial global refinement. */
  unsigned int global_refinement;

  /** The size of this mesh. */
  double h_max;
      
  /** The size of this mesh. */
  double h_min;
      
  /** Dirichlet Boundary Indicators. */
  std::map<char, std::vector<bool> > dirichlet_bc;
      
  /** Neumann Boundary Indicators. */
  std::map<char, std::vector<bool> > neumann_bc;
      
  /** Other Boundary Indicators. */
  std::map<char, std::vector<bool> > other_bc;
    
    /** Parallel data */
  IndexSet locally_relevant_dofs, locally_owned_dofs;
    
  ConditionalOStream pcout;
    
  MPI_Comm& mpi_communicator;
    
private:
  
  std::string fe_name;
  std::vector<std::string> ordering;
  unsigned int map_type;

  std::vector<std::string> d_bc;
  std::vector<std::string> n_bc;
  std::vector<std::string> o_bc;
      
  /** Pointer to the finite element used. */
  SmartPointer<FiniteElement<dim,dim> > fe;

  /** Pointer to the dofhandler used */
    SmartPointer<DoFHandler<dim,dim> > dh;

  /** Pointer to the current triangulation used */
  SmartPointer<parallel::distributed::Triangulation<dim,dim> > tria;

  /** Finite Element Mapping. Various mappings are supported. If the 
      mapping parameter is 0, then cartesian mapping is used. Else Qn, with 
      n the mapping parameter. */
  SmartPointer<Mapping<dim,dim> > mapping;

  public:
      
  /** Number of processes. */
  unsigned int n_mpi_processes;

  /** The id of this process. */
  unsigned int current_mpi_process;

  private:

  /** Number of local dofs. */
  unsigned int n_local_dofs;
  
  /** Number of Blocks. */
  unsigned int number_of_blocks;

  /** Number of cells. */
  unsigned int number_of_cells;
  
  /** Distortion coefficient. */
  double distortion;
  
  /** Refinement strategy. */
  std::string refinement_strategy;

 public:

  /** Bottom fraction of refinement. */
  double bottom_fraction;

  /** Top fraction of refinement. */
  double top_fraction;

 private:

  /** Maximum number of allowed cells. */
  unsigned int max_cells;
    
  /** The wind direction, in case we order the mesh upwind. */
  Point<dim> wind;
};
#endif
