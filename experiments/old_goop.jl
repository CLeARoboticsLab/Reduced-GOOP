# Construct F (equalities), G (inequalities) and z (variables) for the topmost level after recursion.
function construct_kkt(preferences, is_prioritized_constraint, player)
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
			)
			return (; F, z)
		end
	end

	# Recursive call for the next level.
	(; F, z) = construct_kkt(preferences[2:end], is_prioritized_constraint[2:end], player)

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