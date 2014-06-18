//
//  boundary_conditions.h
//  
//
//  Created by Scott Miller on 12/5/11.
//  Copyright 2011 Applied Research Lab, Penn State. All rights reserved.
//

#ifndef _boundary_conditions_h
#define _boundary_conditions_h

    //deal.II includes
#include <deal.II/base/function_parser.h>
#include <deal.II/base/tensor.h>

    // c++ includes
#include <iostream>

template<int dim>
class BoundaryConditions
{
public:
    // (de/con)structors
    BoundaryConditions ();
        // copy constructor
    BoundaryConditions (const BoundaryConditions<dim>& bc);
    ~BoundaryConditions ();
    
    // Member data
    enum BoundaryType { slip,
        fixed,
        zeroGradient,
        symmetry,
        unconstrained,
        fixedNormal,
        totalPressure};
    
    // Member functions
    static inline BoundaryType get_type (std::string boundary_type){
        if (boundary_type == "slip")
            return slip;
        else if (boundary_type == "fixed")
            return fixed;
        else if (boundary_type == "zeroGradient")
            return zeroGradient;
        else if (boundary_type == "symmetry")
            return symmetry;
        else if (boundary_type == "unconstrained")
            return unconstrained;
        else if (boundary_type == "fixedNormal")
            return fixedNormal;
		else if (boundary_type == "totalPressure")
            return totalPressure;
        
        // Need to return something
        return unconstrained;
    }//get_type
    
        // Star values for two phase formulation
    void get_star_values(double& alpha_value,
                        Tensor<1,dim>& velocity_value,
                        double& pressure_value,
                        Vector<double>& prescribed_values,
                        const Point<dim>& normal,
                        double& a_star,
                        Tensor<1,dim>& u_star,
                        double& p_star,
                        double& Da_Di,
                        Tensor<2,dim>& Du_Di,
                        double& Dp_Di,
                         const double& time) const;
    
        // Star values for single phase formulation.  I.e, no alpha field
    void get_star_values(Tensor<1,dim>& velocity_value,
                         double& pressure_value,
                         Vector<double>& prescribed_values,
                         const Point<dim>& normal,
                         Tensor<1,dim>& u_star,
                         double& p_star,
                         Tensor<2,dim>& Du_Di,
                         double& Dp_Di,
                         const double& time) const;
   
        // Conservation variables two phase
//    void get_Mstar_values(double& alpha_value,
//                         Tensor<1,dim>& momentum_value,
//                         double& rho_value,
//                         Vector<double>& prescribed_values,
//                         const Point<dim>& normal,
//                         double& a_star,
//                         Tensor<1,dim>& m_star,
//                         double& r_star,
//                         double& Da_Di,
//                         Tensor<2,dim>& Dm_Di,
//                         double& Dr_Di,
//                         const double& time) const;
    
    unsigned int boundary_id;
    
    std::vector<BoundaryType> type;
    
    FunctionParser<dim> values;
    
};

template <int dim>
BoundaryConditions<dim>::BoundaryConditions ()
:  type(3), 
values (dim+2) // Need to make independent of OnePhase vs TwoPhase
{}

template <int dim>
BoundaryConditions<dim>::BoundaryConditions (const BoundaryConditions<dim>& bc)
:  type(3), 
   values (dim+2)
{
    for(unsigned int i=0; i<2; ++i)
        type[i] = bc.type[i];
}//copy-constructor

template <int dim>
BoundaryConditions<dim>::~BoundaryConditions ()
{}

template <int dim>
void BoundaryConditions<dim>::get_star_values(double& alpha_value,
                     Tensor<1,dim>& velocity_value,
                     double& pressure_value,
                     Vector<double>& prescribed_values,
                     const Point<dim>& normal,
                     double& a_star,
                     Tensor<1,dim>& u_star,
                     double& p_star,
                     double& Da_Di,
                     Tensor<2,dim>& Du_Di,
                     double& Dp_Di,
                     const double &time) const
{
        // Volume fraction:
    if (type[0] == fixed)
    {
        a_star = prescribed_values(0);
        Da_Di = 0.;
    }
    else if (type[0] == symmetry || type[0] == unconstrained)
    {
        a_star = alpha_value;
        Da_Di = 1.;
    }
    else
    {
        std::cout<<"\nNot implemented boundary type for alpha:  "<<type[0]<<std::endl;
        exit(1);
    }
    
        // Velocity
    if (type[1] == fixed)
    {
            // linear ramp
        double factor = 1.;
        
        for (unsigned int d=0; d<dim; ++d)
        {
//            if(time<1.e-6)
//                factor = time/1.e-6;
//            else
//                factor = 1.;
            
            u_star[d] = factor*prescribed_values(d+1);
        }
        
        Du_Di = 0.;
    }
    else if (type[1] == fixedNormal)
    {
            // Small value:
        double eps = 1.e-8;
            // Prescribed: \mathbf{u} \cdot \mathbf{n} = prescribed_values(1)
            
            // Need to check extra stuff in 3D:
        if (dim==2)
        {
            double factor = 1.;
            if(time<1.e-6)
                factor = time/1.e-6;
            
            // Case 1:  nx == 0
            if(std::fabs(normal(0)) < eps)
            {
            u_star[0] = 0.;
            u_star[1] = factor*prescribed_values(1);
            }
                // Case 2:  ny == 0
            else if (std::fabs(normal(1)) < eps)
            {
                u_star[0] = factor*prescribed_values(1);
                u_star[1] = 0.;
            }
            else    // Case 3:  non-coordinate aligned case
            {       // This would work modulo the divide by zero for all cases
                u_star[0] = factor*0.50*prescribed_values(1)/normal(0);
                u_star[1] = factor*0.50*prescribed_values(1)/normal(1);
            }
        }//end dim==2
        else
        {
            std::cout<<"\nfixedNormal BC for velocity not implemented for dim != 2"<<std::endl;
            exit(1);
        }
        
        Du_Di = 0.;
            
    }
    else if (type[1] == symmetry || type[1] == slip)
    {
        double VdotN = velocity_value*Tensor<1,dim>(normal);
        u_star = velocity_value - 2.0*(VdotN*Tensor<1,dim>(normal));
        
            //! Hack:  Is this really a hack?
        u_star = 0.0;
        
        Du_Di = 0.;
        for(unsigned int d=0; d<dim; ++d)
            {
            Du_Di[d][d] += 1.;
            for(unsigned int j=0; j<dim; ++j)
                Du_Di[d][j] -= normal(d)*normal(j);
            }
    }
    else if (type[1] == unconstrained)
    {
        u_star = velocity_value;
        Du_Di = 0.;
        for(unsigned int d=0; d<dim; ++d)
            Du_Di[d][d] =1.;
    }
    else
    {
        std::cout<<"\nNot implemented boundary type for velocity:  "<<type[1]<<std::endl;
        exit(1);
    }
        
        // Pressure
    if (type[2] == fixed)
    {
        p_star = prescribed_values(dim+1);
        Dp_Di = 0.;
    }
    else if (type[2] == symmetry || type[2] == unconstrained)
    {
        p_star = pressure_value;
        Dp_Di = 1.;
    }
    else
    {
        std::cout<<"\nNot implemented boundary type for pressure:  "<<type[2]<<std::endl;
        exit(1);
    }       
        
        
}//get_star_values

/**
 * get_star_values
 *  - No volume fraction here
 *  - This was for single phase testing
 */
template <int dim>
void BoundaryConditions<dim>::get_star_values(Tensor<1,dim>& velocity_value,
                                              double& pressure_value,
                                              Vector<double>& prescribed_values,
                                              const Point<dim>& normal,
                                              Tensor<1,dim>& u_star,
                                              double& p_star,
                                              Tensor<2,dim>& Du_Di,
                                              double& Dp_Di,
                                              const double& time) const
{
        // Velocity
    if (type[0] == fixed)
    {
        for (unsigned int d=0; d<dim; ++d)
            u_star[d] = prescribed_values(d);
        
        Du_Di = 0.;
    }
    else if (type[0] == fixedNormal)
    {
            // Small value:
        double eps = 1.e-8;
            // Prescribed: \mathbf{u} \cdot \mathbf{n} = prescribed_values(1)
            
            // Need to check extra stuff in 3D:
        if (dim==2)
        {
            double factor = 1.;
//             if(time<1.e-6)
//                 factor = time/1.e-6;
            
            // Case 1:  nx == 0
            if(std::fabs(normal(0)) < eps)
            {
            u_star[0] = 0.;
            u_star[1] = factor*prescribed_values(1);
            }
                // Case 2:  ny == 0
            else if (std::fabs(normal(1)) < eps)
            {
                u_star[0] = factor*prescribed_values(1);
                u_star[1] = 0.;
            }
            else    // Case 3:  non-coordinate aligned case
            {       // This would work modulo the divide by zero for all cases
                u_star[0] = factor*0.50*prescribed_values(1)/normal(0);
                u_star[1] = factor*0.50*prescribed_values(1)/normal(1);
            }
        }//end dim==2
        else
        {
            std::cout<<"\nfixedNormal BC for velocity not implemented for dim != 2"<<std::endl;
            exit(1);
        }
        
        Du_Di = 0.;
            
    }
    else if (type[0] == symmetry || type[0] == slip)
    {
        double VdotN = velocity_value*Tensor<1,dim>(normal);
        u_star = velocity_value - 2.0*(VdotN*Tensor<1,dim>(normal));
        
        Du_Di = 0.;
        for(unsigned int d=0; d<dim; ++d)
        {
            Du_Di[d][d] += 1.;
            for(unsigned int j=0; j<dim; ++j)
                Du_Di[d][j] -= normal(d)*normal(j);
        }
        
        u_star = 0.0;
        Du_Di = 0.0;

    }
    else if (type[0] == unconstrained)
    {
        u_star = velocity_value;
        Du_Di = 0.;
        for(unsigned int d=0; d<dim; ++d)
            Du_Di[d][d] =1.;
    }
    else
    {
        std::cout<<"\nNot implemented boundary type for velocity:  "<<type[0]<<std::endl;
        exit(1);
    }
    
        // Pressure
    if (type[1] == fixed)
    {
        p_star = prescribed_values(dim);
        Dp_Di = 0.;
    }
    else if (type[1] == symmetry || type[1] == unconstrained)
    {
        p_star = pressure_value;
        Dp_Di = 1.;
    }
    else if (type[1] == totalPressure)
    {
    	p_star = 841000 - 0.5*1025*(velocity_value*velocity_value);
    }
    else
    {
        std::cout<<"\nNot implemented boundary type for pressure:  "<<type[1]<<std::endl;
        exit(1);
    }       
    
    
}//get_star_values

/*
template <int dim>
void BoundaryConditions<dim>::get_Mstar_values(double& alpha_value,
                                              Tensor<1,dim>& momentum_value,
                                              double& rho_value,
                                              Vector<double>& prescribed_values,
                                              const Point<dim>& normal,
                                              double& a_star,
                                              Tensor<1,dim>& m_star,
                                              double& r_star,
                                              double& Da_Di,
                                              Tensor<2,dim>& Dm_Di,
                                              double& Dr_Di,
                                              const double &time) const
{
        // phase 1 density:
    if (type[0] == fixed)
    {
        a_star = prescribed_values(0);
        Da_Di = 0.;
    }
    else if (type[0] == symmetry || type[0] == unconstrained)
    {
        a_star = alpha_value;
        Da_Di = 1.;
    }
    else
    {
        std::cout<<"\nNot implemented boundary type for alpha:  "<<type[0]<<std::endl;
        exit(1);
    }
    
        // Momentum
    if (type[1] == fixed)
    {
            // linear ramp
        double factor = 1.;
        
        for (unsigned int d=0; d<dim; ++d)
        {
                //            if(time<1.e-6)
                //                factor = time/1.e-6;
                //            else
                //                factor = 1.;
            
            m_star[d] = factor*prescribed_values(d+1);
        }
        
        Dm_Di = 0.;
    }
    else if (type[1] == fixedNormal)
    {
            // Small value:
        double eps = 1.e-8;
            // Prescribed: \mathbf{u} \cdot \mathbf{n} = prescribed_values(1)
        
            // Need to check extra stuff in 3D:
        if (dim==2)
        {
            double factor = 1.;
//            if(time<1.e-6)
//                factor = time/1.e-6;
            
                // Case 1:  nx == 0
            if(std::fabs(normal(0)) < eps)
            {
                m_star[0] = 0.;
                m_star[1] = factor*prescribed_values(1);
            }
                // Case 2:  ny == 0
            else if (std::fabs(normal(1)) < eps)
            {
                m_star[0] = factor*prescribed_values(1);
                m_star[1] = 0.;
            }
            else    // Case 3:  non-coordinate aligned case
            {       // This would work modulo the divide by zero for all cases
                m_star[0] = factor*0.50*prescribed_values(1)/normal(0);
                m_star[1] = factor*0.50*prescribed_values(1)/normal(1);
            }
        }//end dim==2
        else
        {
            std::cout<<"\nfixedNormal BC for velocity not implemented for dim != 2"<<std::endl;
            exit(1);
        }
        
        Dm_Di = 0.;
        
    }
    else if (type[1] == symmetry || type[1] == slip)
    {
        double VdotN = momentum_value*Tensor<1,dim>(normal);
        m_star = momentum_value - (VdotN*Tensor<1,dim>(normal));
        
        Dm_Di = 0.;
        for(unsigned int d=0; d<dim; ++d)
        {
            Dm_Di[d][d] += 1.;
            for(unsigned int j=0; j<dim; ++j)
                Dm_Di[d][j] -= normal(d)*normal(j);
        }
    }
    else if (type[1] == unconstrained)
    {
        m_star = velocity_value;
        Dm_Di = 0.;
        for(unsigned int d=0; d<dim; ++d)
            Dm_Di[d][d] =1.;
    }
    else
    {
        std::cout<<"\nNot implemented boundary type for velocity:  "<<type[1]<<std::endl;
        exit(1);
    }
    
        // Pressure
    if (type[2] == fixed)
    {
        p_star = prescribed_values(dim+1);
        Dp_Di = 0.;
    }
    else if (type[2] == symmetry || type[2] == unconstrained)
    {
        p_star = pressure_value;
        Dp_Di = 1.;
    }
    else
    {
        std::cout<<"\nNot implemented boundary type for pressure:  "<<type[2]<<std::endl;
        exit(1);
    }       
    
    
}//get_Mstar_values
 */



#endif
