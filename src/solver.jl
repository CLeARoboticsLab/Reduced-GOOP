abstract type SolverType end
struct InteriorPoint <: SolverType end

""" Basic interior point solver, based on Nocedal & Wright, ch. 19.
Computes step directions `δz` by solving the relaxed primal-dual system, i.e.
						 ∇F(z; ϵ) δz = -F(z; ϵ).

Given a step direction `δz`, performs a "fraction to the boundary" linesearch,
i.e., for `(x, s)` it chooses step size `α_s` such that
			  α_s = max(α ∈ [0, 1] : s + α δs ≥ (1 - τ) s)
and for `y` it chooses step size `α_s` such that
			  α_y = max(α ∈ [0, 1] : y + α δy ≥ (1 - τ) y).

A typical value of τ is 0.995. Once we converge to ||F(z; \epsilon)|| ≤ ϵ,
we typically decrease ϵ by a factor of 0.1 or 0.2, with smaller values chosen
when the previous subproblem is solved in fewer iterations.

Positional arguments:
	- `mcp::GOOPKKTSystem`: the mixed complementarity problem to solve. #TODO: FIX LATER
	- `θ::AbstractVector{<:Real}`: the parameter vector.

Keyword arguments:
	- `x₀::AbstractVector{<:Real}`: the initial primal variable.
	- `y₀::AbstractVector{<:Real}`: the initial dual variable.
	- `s₀::AbstractVector{<:Real}`: the initial slack variable.
	- `ϵ₀::Real`: the initial relaxation scale.
	- `tol::Real = 1e-4`: the tolerance for the KKT error.
	- `max_inner_iters::Int = 20`: the maximum number of inner iterations.
	- `max_outer_iters::Int = 50`: the maximum number of outer iterations.
	- `tightening_rate::Real = 0.1`: the rate at which to tighten the tolerance.
	- `loosening_rate::Real = 0.5`: the rate at which to loosen the tolerance.
	- `min_stepsize::Real = 1e-2`: the minimum step size for the linesearch.
	- `verbose::Bool = false`: whether to print debug information.
	- `linear_solve_algorithm::LinearSolve.SciMLLinearSolveAlgorithm`: the linear solve algorithm to use. Any solver from `LinearSolve.jl` that can handle nonsquare system can be used.
"""
function solve(
	::InteriorPoint,
	mcp::GOOPKKTSystem,
	θ::AbstractVector{<:Real};
	z₀ = nothing,
	tol = 1e-4,
	ϵ₀ = :auto,
	max_inner_iters = 20,
	max_outer_iters = 50,
	tightening_rate = 0.1,
	loosening_rate = 0.5,
	min_stepsize = 1e-4,
	verbose = false,
	linear_solve_algorithm = LinearSolve.KrylovJL_LSMR(), # LinearSolve.KrylovJL_LSMR(), # KrylovJL_CRAIGMR() for non-square KKT systems
)
	z = @something(z₀, begin
		z = zeros(mcp.variable_dimension)
		z[mcp.preference_slack_dims] .= 1.0
		z[mcp.interior_point_slack_dims] .= 1.0
		z[mcp.inequality_constraint_dual_dims] .= 1.0
		z
	end)

	x = @view z[Not(vcat(mcp.preference_slack_dims, mcp.interior_point_slack_dims, mcp.inequality_constraint_dual_dims))]
	s = @view z[mcp.preference_slack_dims]
	σ = @view z[mcp.interior_point_slack_dims]
	γ = @view z[mcp.inequality_constraint_dual_dims]

	# Set up common memory.
	∇F = mcp.∇F_z!.result_buffer
	F = zeros(mcp.kkt_dimension)
	δz = zeros(mcp.variable_dimension)
	δx = @view δz[Not(vcat(mcp.preference_slack_dims, mcp.interior_point_slack_dims, mcp.inequality_constraint_dual_dims))]
	δs = @view δz[mcp.preference_slack_dims]
	δσ = @view δz[mcp.interior_point_slack_dims]
	δγ = @view δz[mcp.inequality_constraint_dual_dims]

	linsolve = init(LinearProblem(∇F, δz), linear_solve_algorithm, maxiters = 1000)

	# Main solver loop.
	if ϵ₀ === :auto
		is_warmstarted = !isnothing(z₀)
		if is_warmstarted
			ϵ = tol
		else
			ϵ = one(tol)
		end
	else
		ϵ = ϵ₀
	end

	status = :solved
	total_iters = 0
	inner_iters = 1
	outer_iters = 1
	kkt_error = Inf
	while outer_iters < max_outer_iters || iszero(total_iters)
		inner_iters = 1
		status = :solved

		verbose && @info "Outer iteration $(outer_iters): ϵ = $ϵ, kkt_error = $kkt_error"
		# Main.@infiltrate
		while kkt_error > ϵ && inner_iters < max_inner_iters
			total_iters += 1
			# Compute the Newton step.
			# TODO: Can add some adaptive regularization.
			# TODO: use a linear operator with a lazy gradient computation here.
			mcp.F!(F, z; θ, ϵ, η = 0)
			mcp.∇F_z!(∇F, z; θ, ϵ, η = 0)
			@assert all(.!isnan.(F)) "Found NaN in F - aborting!"
			@assert all(.!isnan.(∇F)) "Found NaN in ∇F - aborting!"
			println("condition number of ∇F = $(cond(collect(∇F),2))")
            I_idx = vcat(collect(1:4), collect(7:10), collect(13:16), mcp.variable_dimension) #kkt
            J_idx = vcat(collect(1:4), collect(7:10), collect(13:16), mcp.variable_dimension)
            V = vcat(ones(4), ones(4), ones(4), 0.0)
			linsolve.A = ∇F # + 1e-6 * SparseArrays.sparse(I_idx, J_idx, V) #+ I # adding I to a nonsquare matrix?
			linsolve.b = -F
            # Main.@infiltrate
            # println("inertia of ∇F = $(LinearAlgebra.eigen(Array(∇F'*∇F + 1e-4 * SparseArrays.sparse(I_idx, J_idx, V))))")
			solution = solve!(linsolve)
			if !SciMLBase.successful_retcode(solution) &&
			   (solution.retcode !== SciMLBase.ReturnCode.Default)
				verbose &&
					@warn "Linear solve failed. Exiting prematurely. Return code: $(solution.retcode)"
				status = :failed
				break
			end

			δz .= solution.u

			# Fraction to the boundary linesearch.

			α_σ = fraction_to_the_boundary_linesearch(σ, δσ; tol = min_stepsize)
			α_s = fraction_to_the_boundary_linesearch(s, δs; tol = min_stepsize, max_stepsize = α_σ)
			# α_s = fraction_to_the_boundary_linesearch(s, δs; tol = min_stepsize)
			# α_σ = fraction_to_the_boundary_linesearch(σ, δσ; tol = min_stepsize, max_stepsize = α_s)

			α_γ = fraction_to_the_boundary_linesearch(γ, δγ; tol = min_stepsize)

			println("α_s = $α_s, α_σ = $α_σ, α_γ = $α_γ")

			if isnan(α_s) || isnan(α_γ) || isnan(α_σ)
				verbose && @warn "Linesearch failed. Exiting prematurely."
				status = :failed
				# Main.@infiltrate
				break
			end

			# Update variables accordingly.
			@. x += α_s * δx
			@. s += α_s * δs
			@. σ += α_s * δσ
			@. γ += α_γ * δγ

			# @. x += α_σ * δx
			# @. s += α_σ * δs
			# @. σ += α_σ * δσ
			# @. γ += α_γ * δγ

			kkt_error = norm(F, Inf)

			println("KKT error = $kkt_error")

			inner_iters += 1
		end

		if kkt_error <= ϵ <= tol
			break
		end

		ϵ *= if status === :solved
			1 - exp(-tightening_rate * inner_iters)
		else
			1 + exp(-loosening_rate * inner_iters)
		end
		ϵ = min(ϵ, one(ϵ))
		outer_iters += 1
	end

	if outer_iters == max_outer_iters
		status = :failed
	end

	(; status, z, x, s, σ, γ, kkt_error, ϵ, outer_iters, total_iters)
end

"""Helper function to compute the step size `α` which solves:
				   α* = max(α ∈ [0, 1] : v + α δ ≥ (1 - τ) v).
"""
function fraction_to_the_boundary_linesearch(v, δ; max_stepsize = 1.0, τ = 0.995, decay = 0.5, tol = 1e-4)
	α = max_stepsize
	while any(@. v + α * δ < (1 - τ) * v)
		if α < tol
			return NaN
		end

		α *= decay
	end

	α
end
