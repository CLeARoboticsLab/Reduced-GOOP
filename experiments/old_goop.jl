# Construct F and z (variables) for the topmost level after recursion.
function construct_kkt_old_goop(preferences, is_prioritized_constraint, player)
	level = 1 + length(goop_preferences[player]) - length(preferences)

	# Base level
	if length(preferences) == 1

		f = isnothing(equality_constraints[player]) ? nothing : equality_constraints[player](x, θ)
		g = isnothing(inequality_constraints[player]) ? nothing : inequality_constraints[player](x, θ)
		λ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("λ_$(player)_$(level)"),
			isnothing(f) ? 0 : length(f),
		)
		push!(Λ, λ...)

		γ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("γ_$(player)_$(level)"),
			isnothing(g) ? 0 : length(g),
		)
		push!(Γ, γ...)

		σ = SymbolicTracingUtils.make_variables(
			backend,
			Symbol("σ_$(player)_$(level)"),
			isnothing(g) ? 0 : length(g),
		)
		push!(Σ, σ...)

		if only(is_prioritized_constraint) # TODO for old GOOP
			# Highest priority is a constraint.
			h = only(preferences)(x, θ)

			preference_slack = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("s_$(player)_$(level)"),
				length(h), # get the right dimension
			)

			γₚ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₚ_$(player)_$(level)"),
				length(h),
			)

			γₛ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₛ_$(player)_$(level)"),
				length(h),
			)

			L =
				sum(preference_slack) - γₚ' * (h .+ preference_slack) - γₛ' * preference_slack -
				(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)

			∇L = Symbolics.gradient(L, vcat(x, preference_slack))
			F = Vector{symbolic_type}([∇L; f])
			G = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						isnothing(g) ? nothing : g;
						isnothing(g) ? nothing : γ;
						isnothing(g) ? nothing : θ - γ'*g;
						γₚ; h .+ preference_slack; θ - γₚ'*(h .+ preference_slack);
						γₛ; preference_slack; θ - γₛ' * preference_slack
					],
				),
			)
			z = Vector{symbolic_type}(
				vcat(x, preference_slack, (isnothing(f) ? [] : λ), (isnothing(g) ? [] : γ), γₚ, γₛ),
			)
			return (; F, G, z)
		else
			J = only(preferences)(x, θ)
			L = J - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
			∇L = Symbolics.gradient(L, x)
			F = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L;
						f;
						isnothing(g) ? nothing : g .- σ;
						isnothing(g) ? nothing : σ .* γ .- ϵ;
					],
				),
			)
			z = Vector{symbolic_type}(
				vcat(x, (isnothing(f) ? [] : λ), (isnothing(g) ? [] : γ)),
				# vcat(x, (isnothing(f) ? [] : λ), (isnothing(g) ? [] : γ), (isnothing(g) ? [] : σ)),
			)

			return (; F, z)
		end
	end

	# Recursive call for the next level.
	(; F, z) = construct_kkt_old_goop(preferences[2:end], is_prioritized_constraint[2:end], player)

	J = first(preferences)(x, θ)
	λ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("λ_$(player)_$(level)"),
		length(F),
	)
	push!(Λ, λ...)

	L = J - λ'*F
	∇L = Symbolics.gradient(L, z)
	F̃ = Vector{symbolic_type}([∇L; F])
	(; F = F̃, z = [z; λ])
end

# Older GOOP
function construct_kkt_older_goop(preferences, is_prioritized_constraint, player)
	level = 1 + length(goop_preferences[player]) - length(preferences)

	f = isnothing(equality_constraints[player]) ? nothing : equality_constraints[player](x, θ)
	g = isnothing(inequality_constraints[player]) ? nothing : inequality_constraints[player](x, θ)
	λ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("λ_$(player)_$(level)"),
		isnothing(f) ? 0 : length(f),
	)

	γ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("γ_$(player)_$(level)"),
		isnothing(g) ? 0 : length(g),
	)
	# Base level
	if length(preferences) == 1
		if only(is_prioritized_constraint) # For now, only the innermost is preference constraint.
			# Highest priority is a constraint.
			h = only(preferences)(x, θ)

			preference_slack = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("s_$(player)_$(level)"),
				length(h), # get the right dimension
			)

			γₚ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₚ_$(player)_$(level)"),
				length(h),
			)

			γₛ = SymbolicTracingUtils.make_variables(
				backend,
				Symbol("γₛ_$(player)_$(level)"),
				length(h),
			)

			L =
				sum(preference_slack) - γₚ' * (h .+ preference_slack) - γₛ' * preference_slack -
				(isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)

			∇L = Symbolics.gradient(L, vcat(x, preference_slack))
			F = Vector{symbolic_type}([∇L; f])
			G = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						isnothing(g) ? nothing : g;
						isnothing(g) ? nothing : γ;
						isnothing(g) ? nothing : θ - γ'*g;
						γₚ; h .+ preference_slack; θ - γₚ'*(h .+ preference_slack);
						γₛ; preference_slack; θ - γₛ' * preference_slack
					],
				),
			)
			z = Vector{symbolic_type}(
				vcat(x, preference_slack, (isnothing(f) ? [] : λ), (isnothing(g) ? [] : γ), γₚ, γₛ),
			)
			return (; F, G, z)
		else

			J = only(preferences)(x, θ)
			L = J - (isnothing(f) ? 0 : λ' * f) - (isnothing(g) ? 0 : γ' * g)
			∇L = Symbolics.gradient(L, x)
			F = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						∇L;
						f;
						# isnothing(g) ? nothing : γ .* g .- θ;
					],
				),
			)
			G = Vector{symbolic_type}(
				filter!(
					!isnothing,
					[
						isnothing(g) ? nothing : g;
						isnothing(g) ? nothing : γ;
						isnothing(g) ? nothing : θ - γ'*g;],
				),
			) #ϵ - γ'*g
			z = Vector{symbolic_type}(
				vcat(x, (isnothing(f) ? [] : λ), (isnothing(g) ? [] : γ)),
			)
			return (; F, G, z)
		end
	end

	# Recursive call for the next level.
	(; F, G, z) = construct_kkt_older_goop(preferences[2:end], is_prioritized_constraint[2:end], player)

	J = first(preferences)(x, θ)

	λ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("λ_$(player)_$(level)"),
		length(F),
	)
	γ = SymbolicTracingUtils.make_variables(
		backend,
		Symbol("γ_$(player)_$(level)"),
		length(G),
	)
	L = J - λ'*F - γ'*G
	∇L = Symbolics.gradient(L, z)
	# F̃ = Vector{symbolic_type}([∇L; F; γ .* G .- θ])
	# G̃ = Vector{symbolic_type}([G; γ])
	F̃ = Vector{symbolic_type}([∇L; F])
	G̃ = Vector{symbolic_type}([G; γ; θ - γ'*G])
	# return level == 1 ? (; F = F̃, G = G̃, z = z) : (; F = F̃, G = G̃, z = [z; λ; γ])
	return (; F = F̃, G = G̃, z = [z; λ; γ])
end


using LinearAlgebra

"""
	colspace_issubset(A, B; atol=1e-10, rtol=1e-8, pivot=true) -> Bool

Return `true` iff col(A) ⊆ col(B) numerically.

Implementation: for each column (or all at once), solve the least-squares
problem B*X ≈ A and check that the residuals ‖A - B*X‖ are small.
This uses `\` (QR or pivoted-QR under the hood).
"""
function colspace_issubset(A::AbstractMatrix, B::AbstractMatrix;
	atol::Real = 1e-10, rtol::Real = 1e-8, pivot::Bool = true)
	size(A, 1) == size(B, 1) || throw(ArgumentError("A and B must have the same number of rows"))

	# Empty span(B): only true if A is (numerically) zero
	if size(B, 2) == 0
		return isapprox(A, zero(A); atol = atol, rtol = 0.0)
	end

	# Solve B * X ≈ A in least squares sense.
	# With pivoting for rank-deficient/ill-conditioned B if `pivot=true`.
	X = pivot ? (qr(B, ColumnNorm()) \ A) : (B \ A)

	R = A - B*X                    # residuals in orthogonal complement of span(B)
	# Columnwise mixed tolerance
	# tiny = eps(real(eltype(A)))
	tiny = eps(Float64)
	for j in axes(A, 2)
		aj = view(A, :, j)
		rj = view(R, :, j)
		if norm(rj) > atol + rtol*max(norm(aj), tiny)
			return false
		end
	end
	return true
end
