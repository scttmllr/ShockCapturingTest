/*******************************************************************//**
 * \file userDefinedConstants.h
 * 
 *
 * \brief Namespace for storing various numerical constants such as
 * $\pi$, $2\pi$, $e$, $\epsilon$ values, etc.
 *
 * Purpose:  Encapsulate constants so that they are easy to access and
 *           use in multiple files and projects.
 *
 *   - Provide a few string constants to keep the code clean.
 *   - Provide namespaces for dimensions $1--4$
 *
 *
 *	@author Scott T. Miller
 *	@date  07/31/2010.
 *
 ************************************************************************/
#ifndef _USER_DEFINED_CONSTANTS_
#define _USER_DEFINED_CONSTANTS_

#include <iostream>
#include <string>


//! Namespace for numerical constants
namespace nc{

    //! $\pi$
    static const double pi = 3.14159265358979323846264;
    static const double twoPi = 2.0*pi;

    //! Euler's constant $\e$
    static const double e = 2.71828182845904523536029;

    //! Small and large ``$\epsilon$'' values
    static const double tiny = 1.0e-14;
    static const double huge = 1.0e+14;
    
    static const double epsilon = 1.0e-8;
    static const double neg_epsilon = -1.0*epsilon;
    
    //! Numbers to against allowing for machine precision
    static const double zero = 1.e-8;
    static const double one  = 1. - zero;
    static const double neg_one = -1.0*one;
    static const double neg_zero = -1.0*zero;
    static const double pos_zero = zero;

}//

// String constant namespace
namespace sc{

    //! New line character without buffer flush
    static const std::string nl = "\n";

    //! Warning header:
    static const std::string warning = "*****************************" + nl
                        + "********** WARNING **********" + nl
                        + "*****************************" + nl;
}//namespace sc

//! Some namespaces to define different dimensions
//! that we are working in.  This is to be used as:
//! #include "user_defined_constants.h"
//! using namespace dim3;
//! And then we will have all of the ``variables''
//! defined in that namespace available for use.

namespace dim4 {
    const unsigned int dim = 4;
    const unsigned int codim1 = 3;
    const unsigned int codim2 = 2;
    const unsigned int codim3 = 1;
    const unsigned int codim4 = 0;
}//namespace dim4

namespace dim3 {
    const unsigned int dim = 3;
    const unsigned int codim1 = 2;
    const unsigned int codim2 = 1;
    const unsigned int codim3 = 0;
    const unsigned int codim4 = 0;
}//namespace dim3

namespace dim2 {
    const unsigned int dim = 2;
    const unsigned int codim1 = 1;
    const unsigned int codim2 = 0;
    const unsigned int codim3 = 0;
    const unsigned int codim4 = 0;
}//namespace dim3

namespace dim1 {
    const unsigned int dim = 1;
    const unsigned int codim1 = 0;
    const unsigned int codim2 = 0;
    const unsigned int codim3 = 0;
    const unsigned int codim4 = 0;
}//namespace dim3

#endif