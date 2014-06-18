#include "../include/boundary_indicators.h"
#include <grid/tria_iterator.h>
#include <grid/tria_accessor.h>
#include <fe/fe.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <base/quadrature_lib.h>

#include <grid/tria.h>

#include <fe/fe_dgq.h>
#include <fe/fe_values.h>

template<int dim>						   
void BoundaryIndicators<dim>::set_tria_flags(VectorSpace<dim>* vspace, std::string caseName)
{
        // HSSV 2D case, other cases, etc
    if (caseName == "HSSV 2D" || caseName == "HSSV UNDEX")
        setHSSVTriaFlags (vspace);
    else if (caseName == "Advecting Interface" || caseName == "ShockBubble" || caseName == "UNDEX")
        setAdvectingInterfaceTriaFlags (vspace);
    else if (caseName == "2phaseCyl")
        set2phaseCylTriaFlags (vspace);
    else if (caseName == "Falling drop" || caseName == "Rayleigh Taylor")
        setFallingDropTriaFlags (vspace);
	else if (caseName == "Notional Cav Body")
        setNotionalCavBodyTriaFlags (vspace);
    else if (caseName == "3D Undex")
        set3DUndexTriaFlags (vspace);
    else if (caseName == "Flow Around Cylinder")
    	; // Flags already set when creating "hyper_cube_with_cylindrical_hole"
    else
    {
        std::cout<<sc::warning;
        std::cout<<"\nUnknown case:  "<<caseName
        << "\nAborting..."<<std::endl;
        
        Triangulation<dim>& tria = vspace->get_tria();
        
        typename Triangulation<dim>::active_cell_iterator
        cell = tria.begin_active(),
        endc = tria.end();
        
        unsigned int counter = 0;
        
        for (;cell!=endc; ++cell, ++counter) 
            cell->set_user_index( counter );
    }
    
}//set_tria_flags


template<int dim>						   
void BoundaryIndicators<dim>::setHSSVTriaFlags (VectorSpace<dim>* vspace) 
{
    Triangulation<dim>& tria = vspace->get_tria();
    
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
    
        //std::cout<<"\nIn setTriaFlags!"<<std::endl;		
    
        // Face normal vectors are calculated at 
        // the center of the face:
    Point<dim-1> unit_face_center;
        //Point< 2 > unit_cell_center(0.5,0.5);
    Point<dim> real_cell_center, real_face_center;
    
    if(dim==2)
    {
        unit_face_center(0) = 0.5;
    }
    
    Quadrature<dim-1> single_point_quadrature( unit_face_center );
    MappingQ1<dim> mapping;
    FE_DGQ<dim> fe(0);
    
        // Numbers to check the normal direction against
    double zero = 1.e-8;
    double one  = 1. - zero;
    double neg_one = -1.0*one;
    double neg_zero = -1.0*zero;
    
        //    int mat_id;
    unsigned int counter = 0;
    
    for (;cell!=endc; ++cell, ++counter) 
    {
            //real_cell_center = mapping.transform_unit_to_real_cell( cell, unit_cell_center );
        
            //        userCellData *cell_data = 
            //        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id = 0;
            //		cell->set_material_id( 0 );
        
        cell->set_user_index( counter );
        
            //            std::cout<<"\nNew Cell!, mat_id = "<<reinterpret_cast<userCellData *>(cell->user_pointer())->material_id;
        
        if (cell->at_boundary() )
            for (unsigned int face_no=0; face_no<GeometryInfo<2>::faces_per_cell; ++face_no)
            {
                    // Definition of "face" as an iterator to cell faces.
                typename Triangulation<dim>::face_iterator face = cell->face( face_no );
                
                if (face->at_boundary() )
                {
                        // There might be a way to do this using only triangulation stuff
                    FEFaceValues<dim> fe_values_at_face_center( mapping, fe, 
                                                               single_point_quadrature, 
                                                               UpdateFlags( update_normal_vectors | 
                                                                           update_quadrature_points) );
                    
                    fe_values_at_face_center.reinit( cell, face_no );
                    
                    real_face_center = fe_values_at_face_center.get_quadrature_points()[0];
                    
                        //! There are 8 different boundary flags that I need to set!
                        //! Left, right, and top of the domain
                        //! Cavitator, injectory, body, body ready
                        //! And bottom that is not hssv body
                    if (real_face_center(0) < -1.5 && fe_values_at_face_center.normal_vector(0)(0) < neg_one)
                    {
                            // Left side of the domain
                        face->set_boundary_indicator(1);	// Left
                        
                            //mat_id = int(cell->material_id()) + 1;
                            //cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 1;
                    }
                    else if (real_face_center(0) > 5.4 && fe_values_at_face_center.normal_vector(0)(0) > one)
                    {
                            // Right side of the domain
                        face->set_boundary_indicator(2);
                        
                            //                        mat_id = int(cell->material_id()) + 2;
                            //                        cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 2;
                    }
                    else if (real_face_center(1) > 1.9 && fe_values_at_face_center.normal_vector(0)(1) > one)
                    {
                            // Top side
                        face->set_boundary_indicator(3);
                        
                            //                        mat_id = int(cell->material_id()) + 3;
                            //                        cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 3;
                    }
                    else if (real_face_center(0) < -0.07925 && fe_values_at_face_center.normal_vector(0)(1) < neg_one)
                    {
                            // Bottom, left of cavitator
                        face->set_boundary_indicator(4);
                        
                            //                        mat_id = int(cell->material_id()) + 4;
                            //                        cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 4;
                    }
                    else if (real_face_center(0) > -0.07925 && real_face_center(0) < -0.026975)
                    {
                            // Cavitator
                        face->set_boundary_indicator(5);
                        
                            //                        mat_id = int(cell->material_id()) + 5;
                            //                        cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 5;
                    }
                    else if (real_face_center(0) > -0.026975
                             && real_face_center(0) < -0.0134874
                             && fe_values_at_face_center.normal_vector(0)(0) < neg_zero)
                    {
                            // Injector
                        face->set_boundary_indicator(6);
                        
                            //                        mat_id = int(cell->material_id()) + 6;
                            //                        cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 6;
                    }
                    else if (real_face_center(0) > -0.0134874 && real_face_center(0) < 2.0719292)
                    {
                            // Body
                        face->set_boundary_indicator(7);
                        
                            //                        mat_id = int(cell->material_id()) + 7;
                            //                        cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 7;
                    }
                    else if (real_face_center(0) < (2.072)
                             && real_face_center(0) > (2.071)
                             && fe_values_at_face_center.normal_vector(0)(0) < neg_one)
                    {
                            // Rear of body, vertical
                        face->set_boundary_indicator(8);
                        
                            //                        mat_id = int(cell->material_id()) + 8;
                            //                        cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 8;
                    }
                    else if (real_face_center(0) > 2.0719292
                             && real_face_center(0) < 5.42
                             && fe_values_at_face_center.normal_vector(0)(1) < neg_one)
                    {
                            // Bottom, right of vehicle
                        face->set_boundary_indicator(9);
                        
                            //                        mat_id = int(cell->material_id()) + 9;
                            //                        cell->set_material_id( mat_id );
                            //                        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id += 9;
                    }
                    else
                    {
                        std::cout<<"\nI don't know what boundary indicator to set!"
                        << "\nPoint:  (x,y) = "<<real_face_center
                        << "\nNormal vector:  n = "<<fe_values_at_face_center.normal_vector(0)
                        << std::endl;
                        getchar();
                    }
                    
                }//if-bdry
                 //else 
                 //	face->set_boundary_indicator(7);
                
            }//face			
    }//cell	
}//setHSSVTriaFlags

template<int dim>						   
void BoundaryIndicators<dim>::setNotionalCavBodyTriaFlags (VectorSpace<dim>* vspace) 
{
    Triangulation<dim>& tria = vspace->get_tria();
    
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
    
        //std::cout<<"\nIn setTriaFlags!"<<std::endl;		
    
        // Face normal vectors are calculated at 
        // the center of the face:
    Point<dim-1> unit_face_center;
        //Point< 2 > unit_cell_center(0.5,0.5);
    Point<dim> real_cell_center, real_face_center;
    
    if(dim==2)
    {
        unit_face_center(0) = 0.5;
    }
    
    Quadrature<dim-1> single_point_quadrature( unit_face_center );
    MappingQ1<dim> mapping;
    FE_DGQ<dim> fe(0);
    
        //    int mat_id;
    unsigned int counter = 0;
    
    Point<dim> normal;
    
    for (;cell!=endc; ++cell, ++counter) 
    {   
        cell->set_user_index( counter );
        
        if (cell->at_boundary() )
            for (unsigned int face_no=0; face_no<GeometryInfo<2>::faces_per_cell; ++face_no)
            {
                    // Definition of "face" as an iterator to cell faces.
                typename Triangulation<dim>::face_iterator face = cell->face( face_no );
                
                if (face->at_boundary() )
                {
                        // There might be a way to do this using only triangulation stuff
                    FEFaceValues<dim> fe_values_at_face_center( mapping, fe, 
                                                               single_point_quadrature, 
                                                               UpdateFlags( update_normal_vectors | 
                                                                           update_quadrature_points) );
                    
                    fe_values_at_face_center.reinit( cell, face_no );
                    
                    real_face_center = fe_values_at_face_center.get_quadrature_points()[0];
                    
                    normal = fe_values_at_face_center.normal_vector(0);
                    
                        //! There are 8 different boundary flags that I need to set!
                        //! Left, right, and top of the domain
                        //! Cavitator, injectory, body, body ready
                        //! And bottom that is not hssv body
                    if (real_face_center(0) < (-10.+nc::zero) 
                        && normal(0) < nc::neg_one)
                    {
                            // Left side of the domain
                        face->set_boundary_indicator(1);	// Left
                        
                    }
                    else if (real_face_center(0) > (30.1+nc::neg_zero) 
                             && normal(0) > nc::one)
                    {
                            // Right side of the domain
                        face->set_boundary_indicator(2);
                    }
                    else if (real_face_center(1) > (30.0+nc::neg_one )
                             && normal(1) > nc::one)
                    {
                            // Top side
                        face->set_boundary_indicator(3);
                    }
                    else if (real_face_center(0) < nc::zero 
                             && normal(1) < nc::neg_one)
                    {
                            // Bottom, left of cavitator
                        face->set_boundary_indicator(4);
                    }
                    else if ( (real_face_center(0) < (0.10 + nc::zero) 
                               && normal(1) < nc::neg_one)
                             || (real_face_center(0) > nc::neg_zero 
                                 && normal(0) > nc::one) )
                    {
                            // Cavitator
                        face->set_boundary_indicator(5);
                    }
                    else if (real_face_center(0) < (0.10 + nc::zero)
                             && real_face_center(0) > (0.10 - nc::zero)
                             && normal(0) < nc::neg_one)
                    {
                            // Injector
                        face->set_boundary_indicator(6);
                    }
                    else if (real_face_center(0) > 0.10 
                             && real_face_center(0) < 10.1)
                    {
                            // Body
                        face->set_boundary_indicator(7);
                    }
                    else if (real_face_center(0) < (10.1+nc::zero)
                             && real_face_center(0) > (10.1-nc::zero)
                             && normal(0) < nc::neg_one)
                    {
                            // Rear of body, vertical
                        face->set_boundary_indicator(8);
                    }
                    else if (real_face_center(0) > 10.1
                             && normal(1) < nc::neg_one)
                    {
                            // Bottom, right of vehicle
                        face->set_boundary_indicator(9);
                    }
                    else
                    {
                        std::cout<<"\nI don't know what boundary indicator to set!"
                        << "\nPoint:  (x,y) = "<<real_face_center
                        << "\nNormal vector:  n = "<<fe_values_at_face_center.normal_vector(0)
                        << std::endl;
                        getchar();
                    }
                    
                }//if-bdry
                 //else 
                 //	face->set_boundary_indicator(7);
                
            }//face			
    }//cell	
}//setNotionalCavBodyTriaFlags

template<int dim>						   
void BoundaryIndicators<dim>::set2phaseCylTriaFlags (VectorSpace<dim>* vspace) 
{
    Triangulation<dim>& tria = vspace->get_tria();
    
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();		
    
        // Face normal vectors are calculated at 
        // the center of the face:
    Point<dim-1> unit_face_center(0.5);
        //Point< 2 > unit_cell_center(0.5,0.5);
    Point<dim> real_cell_center, real_face_center;
    
    Quadrature<dim-1> single_point_quadrature( unit_face_center );
    MappingQ1<dim> mapping;
    FE_DGQ<dim> fe(0);
    
    if(dim==2)
        unit_face_center(0) = 0.5;
    
        //    int mat_id;
    unsigned int counter = 0;
    
    for (;cell!=endc; ++cell, ++counter) 
    {
            //real_cell_center = mapping.transform_unit_to_real_cell( cell, unit_cell_center );
        
            //        userCellData *cell_data = 
            //        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id = 0;
            //		cell->set_material_id( 0 );
        
        cell->set_user_index( counter );
        
            //            std::cout<<"\nNew Cell!, mat_id = "<<reinterpret_cast<userCellData *>(cell->user_pointer())->material_id;
        
        if (cell->at_boundary() )
            for (unsigned int face_no=0; face_no<GeometryInfo<2>::faces_per_cell; ++face_no)
            {
                    // Definition of "face" as an iterator to cell faces.
                typename Triangulation<dim>::face_iterator face = cell->face( face_no );
                
                if (face->at_boundary() )
                {
                        // There might be a way to do this using only triangulation stuff
                    FEFaceValues<dim> fe_values_at_face_center( mapping, fe, 
                                                               single_point_quadrature, 
                                                               UpdateFlags( update_normal_vectors | 
                                                                           update_quadrature_points) );
                    
                    fe_values_at_face_center.reinit( cell, face_no );
                    
                    real_face_center = fe_values_at_face_center.get_quadrature_points()[0];
                    
                        //! There are 6 different boundary flags that I need to set!
                        //! Leftx2, right, top, bottom of the domain
                        //! Cylinder is none of the above!
                    if (real_face_center(0) < -2. 
                        && real_face_center(1) < 0.0 
                        && fe_values_at_face_center.normal_vector(0)(0) < nc::neg_one)
                    {
                            // Left side of the domain, water
                        face->set_boundary_indicator(1);	// Left
                    }
                    else if (real_face_center(0) < -2.
                             && real_face_center(1) >= 0.0 
                             && fe_values_at_face_center.normal_vector(0)(0) < nc::neg_one)
                    {
                            // Left side of the domain, air
                        face->set_boundary_indicator(2);	// Left
                    }
                    else if (real_face_center(0) > 10. && fe_values_at_face_center.normal_vector(0)(0) > nc::one)
                    {
                            // Right side of the domain
                        face->set_boundary_indicator(5);
                    }
                    else if (real_face_center(1) > 2. && fe_values_at_face_center.normal_vector(0)(1) > nc::one)
                    {
                            // Top side
                        face->set_boundary_indicator(4);
                    }
                    else if (real_face_center(0) < -2. && fe_values_at_face_center.normal_vector(0)(1) < nc::neg_one)
                    {
                            // Bottom
                        face->set_boundary_indicator(3);
                    }
                    else 
                    {
                            // Cylinder
                        face->set_boundary_indicator(6);
                    }
                    
                }//if-bdry
                 //else 
                 //	face->set_boundary_indicator(7);
                
            }//face			
    }//cell	
}//set2phaseCylTriaFlags

template<int dim>						   
void BoundaryIndicators<dim>::setFallingDropTriaFlags (VectorSpace<dim>* vspace) 
{
    Triangulation<dim>& tria = vspace->get_tria();
    
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();		
    
        // Face normal vectors are calculated at 
        // the center of the face:
    Point<dim-1> unit_face_center(0.5);
        //Point< 2 > unit_cell_center(0.5,0.5);
    Point<dim> real_cell_center, real_face_center;
    
    Quadrature<dim-1> single_point_quadrature( unit_face_center );
    MappingQ1<dim> mapping;
    FE_DGQ<dim> fe(0);
    
    if(dim==2)
        unit_face_center(0) = 0.5;
    
        //    int mat_id;
    unsigned int counter = 0;
    
    for (;cell!=endc; ++cell, ++counter) 
    {
            //real_cell_center = mapping.transform_unit_to_real_cell( cell, unit_cell_center );
        
            //        userCellData *cell_data = 
            //        reinterpret_cast<userCellData *>(cell->user_pointer())->material_id = 0;
            //		cell->set_material_id( 0 );
        
        cell->set_user_index( counter );
        
            //            std::cout<<"\nNew Cell!, mat_id = "<<reinterpret_cast<userCellData *>(cell->user_pointer())->material_id;
        
        if (cell->at_boundary() )
            for (unsigned int face_no=0; face_no<GeometryInfo<2>::faces_per_cell; ++face_no)
            {
                    // Definition of "face" as an iterator to cell faces.
                typename Triangulation<dim>::face_iterator face = cell->face( face_no );
                
                if (face->at_boundary() )
                {
                        // There might be a way to do this using only triangulation stuff
                    FEFaceValues<dim> fe_values_at_face_center( mapping, fe, 
                                                               single_point_quadrature, 
                                                               UpdateFlags( update_normal_vectors | 
                                                                           update_quadrature_points) );
                    
                    fe_values_at_face_center.reinit( cell, face_no );
                    
                    real_face_center = fe_values_at_face_center.get_quadrature_points()[0];
                    
                        // 3 boundaries:  left, top, bottom.
                        // All have some BC:  no flux.  Set all to 1
                    face->set_boundary_indicator(1);
                    
                        // Right boundary is symmetry:  set to 2
                    if(fe_values_at_face_center.normal_vector(0)(0) > 0.5)
                        face->set_boundary_indicator(2);
                    
                }//if-bdry
                
            }//face			
    }//cell	
}//set_falling_drop

template<int dim>
void BoundaryIndicators<dim>::set3DUndexTriaFlags (VectorSpace<dim>* vspace)
{
    Triangulation<dim>& tria = vspace->get_tria();
    
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
    
    for (;cell!=endc; ++cell)
    {
        if (cell->at_boundary() )
            for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
                    // Definition of "face" as an iterator to cell faces.
                typename Triangulation<dim>::face_iterator face = cell->face( face_no );
                
                if (face->at_boundary() )
                {
                    face->set_boundary_indicator(0);
                }//if-bdry
                
            }//face
    }//cell
}//set3DUndexTriaFlags

template<int dim>						   
void BoundaryIndicators<dim>::setAdvectingInterfaceTriaFlags (VectorSpace<dim>* vspace) 
{
    Triangulation<dim>& tria = vspace->get_tria();
    
    typename Triangulation<dim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();		
    
        // Face normal vectors are calculated at 
        // the center of the face:
    Point<dim-1> unit_face_center(0.5);
        //Point< 2 > unit_cell_center(0.5,0.5);
    Point<dim> real_cell_center, real_face_center;
    
    Quadrature<dim-1> single_point_quadrature( unit_face_center );
    MappingQ1<dim> mapping;
    FE_DGQ<dim> fe(0);
    
    if(dim==2)
        unit_face_center(0) = 0.5;
    
        //    int mat_id;
    unsigned int counter = 0;
    
    for (;cell!=endc; ++cell, ++counter) 
    {
        cell->set_user_index( counter );
        
        if (cell->at_boundary() )
            for (unsigned int face_no=0; face_no<GeometryInfo<2>::faces_per_cell; ++face_no)
            {
                    // Definition of "face" as an iterator to cell faces.
                typename Triangulation<dim>::face_iterator face = cell->face( face_no );
                
                if (face->at_boundary() )
                {
                        // There might be a way to do this using only triangulation stuff
                    FEFaceValues<dim> fe_values_at_face_center( mapping, fe, 
                                                               single_point_quadrature, 
                                                               UpdateFlags( update_normal_vectors | 
                                                                           update_quadrature_points) );
                    
                    fe_values_at_face_center.reinit( cell, face_no );
                    
                    real_face_center = fe_values_at_face_center.get_quadrature_points()[0];
                    
                    
                    if (fe_values_at_face_center.normal_vector(0)(0) < nc::neg_one)
                    {
                            // Left side of the domain
                        face->set_boundary_indicator(0);	// Left
                    }
                    else if (fe_values_at_face_center.normal_vector(0)(0) > nc::one)
                    {
                            // Left side of the domain
                        face->set_boundary_indicator(1);	// Right
                    }
                    else if (fe_values_at_face_center.normal_vector(0)(1) < nc::neg_one)
                    {
                            // Left side of the domain
                        face->set_boundary_indicator(2);	// Bottom
                    }
                    else if (fe_values_at_face_center.normal_vector(0)(1) > nc::one)
                    {
                            // Left side of the domain
                        face->set_boundary_indicator(3);	// Top
                    }
                    else
                    {
                        std::cout<<"\nFunction:  setAdvectingInterfaceTriaFlags()";
                        std::cout<<"\nI don't know what boundary indicator to set!"
                        << "\nPoint:  (x,y) = "<<real_face_center
                        << "\nNormal vector:  n = "<<fe_values_at_face_center.normal_vector(0)
                        << std::endl;
                        getchar();
                    }
                    
                }//if-bdry
                 //else 
                 //	face->set_boundary_indicator(7);
                
            }//face			
    }//cell	
}//setAdvectingInterfaceTriaFlags

template class BoundaryIndicators<deal_II_dimension>;
