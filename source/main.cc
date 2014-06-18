#include <deal.II/base/logstream.h>
#include "../include/base.h"

int main (int argc, char* argv[]) 
{
  std::string prm_file_name;

  if(argc < 2) 
    {
    dealii::deallog << "No Parameter File Specified!!!" << std::endl;
    dealii::deallog << "Using Default Filename: Parameters.prm" << std::endl;
    prm_file_name = "parameters.prm";
    } 
  else 
    {
    prm_file_name = argv[1];
    }
 
    Utilities::System::MPI_InitFinalize mpi_initialization(argc, argv);
    
        {  
        Base<deal_II_dimension> base;
        base.run (prm_file_name);
        }
  
return 0;
}
