
#include "Halide.h"

using namespace Halide;

namespace {

// Generator class for BLAS gemm operations.
    template<class T>
    class GEMMGenerator :
            public Generator<GEMMGenerator<T>> {
    public:
      typedef Generator<GEMMGenerator<T>> Base;
      using Base::target;
      using Base::get_target;
      using Base::natural_vector_size;

      // Generator Param
      GeneratorParam<bool> transpose_A_ = {"transpose_A", false};
      GeneratorParam<bool> transpose_B_ = {"transpose_B", false};

      // Standard ordering of parameters in GEMM functions.
      Param<T>   a_ = {"a", 1.0};
      ImageParam A_ = {type_of<T>(), 2, "A"};
      ImageParam B_ = {type_of<T>(), 2, "B"};
      Param<T>   b_ = {"b", 1.0};
      ImageParam C_ = {type_of<T>(), 2, "C"};

      Var i, j;

      Func build() {
        Func result("result");
        // Matrices are interpreted as column-major by default. The
        // transpose GeneratorParams are used to handle cases where
        // one or both is actually row major.
        const Expr num_rows = A_.width();
        const Expr num_cols = B_.height();
        const Expr sum_size = A_.height();

        // If they're both transposed, then reverse the order and transpose the result instead.
        bool transpose_AB = false;
        if ((bool)transpose_A_ && (bool)transpose_B_) {
          std::swap(A_, B_);
          transpose_A_.set(false);
          transpose_B_.set(false);
          transpose_AB = true;
        }

        Func A("A"), B("B"), Btmp("Btmp"), Atmp("Atmp");

        //TODO: Swizzle A for better memory order in the inner loop.

        Atmp(i, j) = A_(i, j);
        if(transpose_A_){
          A(i, j) = Atmp(j, i);
        }else{
          A(i, j) = Atmp(i, j);
        }

        Btmp(i, j) = B_(i, j);
        if (transpose_B_) {
          B(i, j) = Btmp(j, i);
        } else {
          B(i, j) = Btmp(i, j);
        }

        Var k("k");
        Func prod;
        // Express all the products we need to do a matrix multiply as a 3D Func.
        prod(k, i, j) = A(i, k) * B(k, j);

        // Reduce the products along k.
        Func AB("AB");
        RDom rv(0, sum_size);
        AB(i, j) += prod(rv, i, j);

        Func ABt("ABt");
        if (transpose_AB) {
          // Transpose A*B if necessary.
          ABt(i, j) = AB(j, i);
        } else {
          ABt(i, j) = AB(i, j);
        }

        // Do the part that makes it a 'general' matrix multiply.
        result(i, j) = (a_ * ABt(i, j) + b_ * C_(i, j));

        // TODO: no scheduling for now, need optimization

        return result;
      }
    };

    RegisterGenerator<GEMMGenerator<float>>    register_sgemm("sgemm");
    RegisterGenerator<GEMMGenerator<double>>   register_dgemm("dgemm");

}  // namespace
