using SymbolicTracingUtils

using  QuasiGOOP

backend = SymbolicsBackend()
x = make_variables(backend, :x, 3)
println("x = ", x)

f_symbolic = x[1]^2 + x[2]*x[3]
f_callable = build_function([f_symbolic], x; in_place = false)
result = f_callable([1.0, 2.0, 3.0])
println("result = ", result)
grad_symbolic = gradient(f_symbolic, x)
println("grad_symbolic = ", grad_symbolic)
grad_callable! = let 
    _grad_callable! = build_function(grad_symbolic, x; in_place = true)
    (val, x) -> _grad_callable!(val, x)
end
F_val = zeros(6)
grad_callable!(F_val, [1.0, 2.0, 3.0]) # in-place computation
println("F_val = ", F_val) # F_val =is modified to [2.0, 3.0, 0.0, 0.0, 0.0, 0.0]

#######
f_vector = [x[1]^2, x[2]*x[3]]
jac_symbolic = jacobian(f_vector, x)  # Dense Jacobian
println("jac_symbolic = ", jac_symbolic)
sparse_jac_symbolic = sparse_jacobian(f_vector, x)  # Sparse Jacobian
println("tensor_symbolic = ", tensor_symbolic)
# check dimensions
println("ndims(jac_symbolic) = ", ndims(jac_symbolic))
println("ndims(sparse_jac_symbolic) = ", ndims(sparse_jac_symbolic))

# Example 
x = make_variables(backend, :x, 6)
J_symbolic1 = sum(x[i]^2 for i in 1:3) # some cost function
J_symbolic2 = sum(x[i]^2 for i in 4:only(size(x))) 
f_vector = [x[1]^2, x[2]*x[3]]
λ1 = make_variables(backend, :λ1, only(size(f_vector)))
λ2 = make_variables(backend, :λ2, only(size(x)))

Lagrangian_symbolic = J_symbolic1 - (gradient(J_symbolic2, x) - jacobian(f_vector, x)' * λ1)'  * λ2
println("Lagrangian_symbolic = ", Lagrangian_symbolic)
