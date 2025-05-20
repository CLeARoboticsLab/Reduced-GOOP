using SymbolicTracingUtils: SymbolicTracingUtils

using QuasiGOOP

backend = SymbolicTracingUtils.SymbolicsBackend()
x = SymbolicTracingUtils.make_variables(backend, :x, 3)
println("x = ", x)

f_symbolic = x[1]^2 + x[2]*x[3]
f_callable = SymbolicTracingUtils.build_function([f_symbolic], x; in_place = false)
result = f_callable([1.0, 2.0, 3.0])
println("result = ", result)
grad_symbolic = SymbolicTracingUtils.gradient(f_symbolic, x)
println("grad_symbolic = ", grad_symbolic)
grad_callable! = let
    _grad_callable! =
        SymbolicTracingUtils.build_function(grad_symbolic, x; in_place = true)
    (val, x) -> _grad_callable!(val, x)
end
F_val = zeros(6)
grad_callable!(F_val, [1.0, 2.0, 3.0]) # in-place computation
println("F_val = ", F_val) # F_val =is modified to [2.0, 3.0, 0.0, 0.0, 0.0, 0.0]

#######
f_vector = [x[1]^2, x[2]*x[3]]
jac_symbolic = SymbolicTracingUtils.jacobian(f_vector, x)  # Dense Jacobian
println("jac_symbolic = ", jac_symbolic)
sparse_jac_symbolic = SymbolicTracingUtils.sparse_jacobian(f_vector, x)  # Sparse Jacobian
# check dimensions
println("ndims(jac_symbolic) = ", SymbolicTracingUtils.ndims(jac_symbolic))
println("ndims(sparse_jac_symbolic) = ", SymbolicTracingUtils.ndims(sparse_jac_symbolic))

# Example 
x = SymbolicTracingUtils.make_variables(backend, :x, 6)
J_symbolic1 = sum(x[i]^2 for i in 1:3) # some cost function
J_symbolic2 = sum(x[i]^2 for i in 4:only(size(x)))
f_vector = [x[1]^2, x[2]*x[3]]
λ1 = SymbolicTracingUtils.make_variables(backend, :λ1, only(size(f_vector)))
λ2 = SymbolicTracingUtils.make_variables(backend, :λ2, only(size(x)))

Lagrangian_symbolic =
    J_symbolic1 -
    (
        SymbolicTracingUtils.gradient(J_symbolic2, x) -
        SymbolicTracingUtils.jacobian(f_vector, x)' * λ1
    )' * λ2
println("Lagrangian_symbolic = ", Lagrangian_symbolic)
