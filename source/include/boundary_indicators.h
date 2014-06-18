#ifndef _BOUNDARY_INDICATORS_H
#define _BOUNDARY_INDICATORS_H

    // local/project headers
#include "vector_space.h"
#include "user_defined_constants.h"

    // c++
#include <string>

using namespace dealii;

/** 
  Static class to set boundary indicators.
    - No constructors
*/
template <int dim>
class BoundaryIndicators
{
    public:
      
        static void set_tria_flags(VectorSpace<dim>* vspace, 
                                   std::string caseName);


    private:
    
        static void setHSSVTriaFlags               (VectorSpace<dim>* vspace);
        static void set2phaseCylTriaFlags          (VectorSpace<dim>* vspace);
        static void setAdvectingInterfaceTriaFlags (VectorSpace<dim>* vspace);
        static void setFallingDropTriaFlags        (VectorSpace<dim>* vspace);
        static void setNotionalCavBodyTriaFlags    (VectorSpace<dim>* vspace);
        static void set3DUndexTriaFlags    (VectorSpace<dim>* vspace);
    
};

#endif
