#ifndef FSI_UTILITIES_H
#define FSI_UTILITIES_H

#include <sys/stat.h>

namespace FSIutilities
{

inline void createDirectory(const char* dir)
{
#if defined(_MSC_VER)
    struct _stat dir_stat;
    if(_stat(dir, &dir_stat) == -1)//the directory doesn't exist
           _mkdir(dir);
#else
    struct stat dir_stat;
    if(stat(dir, &dir_stat) == -1)//the directory doesn't exist
           mkdir(dir, 0777);
#endif
}//end-createDirectory

/**
 * Inner product of two second order tensors, returns a double
 * @relates Tensor
 */
template <int dim>
inline double operator * (const Tensor<2,dim> &src1,
                          const Tensor<2,dim> &src2)
{
    double dest = 0;
    for (unsigned int i=0; i<dim; ++i)
        for (unsigned int j=0; j<dim; ++j)
            dest += src1[i][j] * src2[i][j];
    return dest;
}

/**
 * Tensor Inner Product a tensor of rank 2 with a tensor of rank 2. The result is
 * <tt>dest = sum_ij src1[i][j] src2[i][j]</tt>.
 * @relates Tensor
 */
template <int dim>
inline double TensorInnerProduct (const Tensor<2,dim> &src1,
                                  const Tensor<2,dim> &src2)
{
    double dest=0;
    for (unsigned int i=0; i<dim; ++i)
        for (unsigned int j=0; j<dim; ++j)
                dest += src1[i][j] * src2[i][j];
    return dest;
}


/**
 * Takes the product of a tensor of rank 1,dim and a vector with dim components. The result is
 * <tt>dest = sum_i src1[i] src2(i)</tt>.
 * @relates Tensor
 */
template <int dim>
inline double TensorVectorProduct (const Tensor<1,dim> &src1,
                                   const Vector<double> &src2)
{
    Assert (src2.size() == dim, ExcDimensionMismatch (src2.size(), dim));
    double dest = 0;
    for (unsigned int i=0; i<dim; ++i)
        dest += src1[i] * src2(i);
    return dest;
}

/**
 * Takes the product of  a vector with dim components and a tensor of rank 1,dim. The result is
 * <tt>dest = sum_i src1(i) src2[i]</tt>.
 * @relates Tensor
 */
template <int dim>
inline double TensorVectorProduct (const Vector<double> &src1,
                                   const Tensor<1,dim> &src2)
{
    Assert (src1.size() == dim, ExcDimensionMismatch (src2.size(), dim));
    double dest = 0;
    for (unsigned int i=0; i<dim; ++i)
        dest += src1(i) * src2[i];
    return dest;
}

}//namespace

#endif
