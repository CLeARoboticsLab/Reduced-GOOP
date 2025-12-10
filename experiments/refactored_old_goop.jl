# === OLD GOOP KKT builder (refactored to avoid globals) ===

"""
    construct_kkt_old_goop!(
        backend,
        preferences,
        is_prioritized_constraint,
        player,
        total_levels,
        x, θ, ϵ,
        equality_constraints,
        inequality_constraints,
        Λ, Γ, Σ, s,
    ) -> (F, z)

Recursively construct the old-GOOP KKT system for levels 2..K.

All mutable state (Λ, Γ, Σ, s) is passed in explicitly, so there are
no global variables.
"""
function construct_kkt_old_goop!(
    backend,
    preferences,
    is_prioritized_constraint,
    player::Int,
    total_levels::Int,
    x,
    θ,
    ϵ,
    equality_constraints,
    inequality_constraints,
    Λ::Vector,
    Γ::Vector,
    Σ::Vector,
    s::Vector,
)
    symbolic_type = eltype(x)
    level = 1 + total_levels - length(preferences)

    # Base level
    if length(preferences) == 1
        f = isnothing(equality_constraints[player]) ? nothing :
            equality_constraints[player](x, θ)
        g = isnothing(inequality_constraints[player]) ? nothing :
            inequality_constraints[player](x, θ)

        λ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("λ_$(player)_$(level)"),
            isnothing(f) ? 0 : length(f),
        )
        append!(Λ, λ)

        γ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("γ_$(player)_$(level)"),
            isnothing(g) ? 0 : length(g),
        )
        append!(Γ, γ)

        σ = SymbolicTracingUtils.make_variables(
            backend,
            Symbol("σ_$(player)_$(level)"),
            isnothing(g) ? 0 : length(g),
        )
        append!(Σ, σ)

        if only(is_prioritized_constraint)
            # Highest priority is a constraint.
            h = only(preferences)(x, θ)

            preference_slack = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("s_$(player)_$(level)"),
                length(h),
            )
            append!(s, preference_slack)  # track slacks globally via `s`

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
                sum(preference_slack) -
                γₚ' * (h .+ preference_slack) -
                γₛ' * preference_slack -
                (isnothing(f) ? 0 : λ' * f) -
                (isnothing(g) ? 0 : γ' * g)

            ∇L = Symbolics.gradient(L, vcat(x, preference_slack))

            F = Vector{symbolic_type}([∇L; f])

            G = Vector{symbolic_type}(
                filter!(
                    !isnothing,
                    [
                        isnothing(g) ? nothing : g;
                        isnothing(g) ? nothing : γ;
                        isnothing(g) ? nothing : θ - γ' * g;
                        γₚ;
                        h .+ preference_slack;
                        θ - γₚ' * (h .+ preference_slack);
                        γₛ;
                        preference_slack;
                        θ - γₛ' * preference_slack;
                    ],
                ),
            )

            z = Vector{symbolic_type}(
                vcat(x,
                     preference_slack,
                     (isnothing(f) ? [] : λ),
                     (isnothing(g) ? [] : γ),
                     γₚ,
                     γₛ),
            )

            return (; F, G, z)
        else
            # Standard objective
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
                vcat(x,
                     (isnothing(f) ? [] : λ),
                     (isnothing(g) ? [] : γ)),
            )

            return (; F, z)
        end
    end

    # Recursive call for the next level (k+1..K)
    lower = construct_kkt_old_goop!(
        backend,
        preferences[2:end],
        is_prioritized_constraint[2:end],
        player,
        total_levels,
        x,
        θ,
        ϵ,
        equality_constraints,
        inequality_constraints,
        Λ,
        Γ,
        Σ,
        s,
    )

    F_lower = lower.F
    z_lower = lower.z

    J = first(preferences)(x, θ)

    λ = SymbolicTracingUtils.make_variables(
        backend,
        Symbol("λ_$(player)_$(level)"),
        length(F_lower),
    )
    append!(Λ, λ)

    L = J - λ' * F_lower
    ∇L = Symbolics.gradient(L, z_lower)

    F̃ = Vector{symbolic_type}([∇L; F_lower])
    z̃ = [z_lower; λ]

    return (; F = F̃, z = z̃)
end

# === Older GOOP variant, refactored similarly  ===

function construct_kkt_older_goop!(
    backend,
    preferences,
    is_prioritized_constraint,
    player::Int,
    total_levels::Int,
    x,
    θ,
    equality_constraints,
    inequality_constraints,
)
    symbolic_type = eltype(x)
    level = 1 + total_levels - length(preferences)

    f = isnothing(equality_constraints[player]) ? nothing :
        equality_constraints[player](x, θ)
    g = isnothing(inequality_constraints[player]) ? nothing :
        inequality_constraints[player](x, θ)

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
        if only(is_prioritized_constraint)
            # Highest priority is a constraint.
            h = only(preferences)(x, θ)

            preference_slack = SymbolicTracingUtils.make_variables(
                backend,
                Symbol("s_$(player)_$(level)"),
                length(h),
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
                sum(preference_slack) -
                γₚ' * (h .+ preference_slack) -
                γₛ' * preference_slack -
                (isnothing(f) ? 0 : λ' * f) -
                (isnothing(g) ? 0 : γ' * g)

            ∇L = Symbolics.gradient(L, vcat(x, preference_slack))

            F = Vector{symbolic_type}([∇L; f])

            G = Vector{symbolic_type}(
                filter!(
                    !isnothing,
                    [
                        isnothing(g) ? nothing : g;
                        isnothing(g) ? nothing : γ;
                        isnothing(g) ? nothing : θ - γ' * g;
                        γₚ;
                        h .+ preference_slack;
                        θ - γₚ' * (h .+ preference_slack);
                        γₛ;
                        preference_slack;
                        θ - γₛ' * preference_slack;
                    ],
                ),
            )

            z = Vector{symbolic_type}(
                vcat(x,
                     preference_slack,
                     (isnothing(f) ? [] : λ),
                     (isnothing(g) ? [] : γ),
                     γₚ,
                     γₛ),
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
                    ],
                ),
            )

            G = Vector{symbolic_type}(
                filter!(
                    !isnothing,
                    [
                        isnothing(g) ? nothing : g;
                        isnothing(g) ? nothing : γ;
                        isnothing(g) ? nothing : θ - γ' * g;
                    ],
                ),
            )

            z = Vector{symbolic_type}(
                vcat(x,
                     (isnothing(f) ? [] : λ),
                     (isnothing(g) ? [] : γ)),
            )
            return (; F, G, z)
        end
    end

    # Recursive call for next level
    lower = construct_kkt_older_goop!(
        backend,
        preferences[2:end],
        is_prioritized_constraint[2:end],
        player,
        total_levels,
        x,
        θ,
        equality_constraints,
        inequality_constraints,
    )

    F_lower = lower.F
    G_lower = lower.G
    z_lower = lower.z

    J = first(preferences)(x, θ)

    λ_level = SymbolicTracingUtils.make_variables(
        backend,
        Symbol("λ_$(player)_$(level)"),
        length(F_lower),
    )
    γ_level = SymbolicTracingUtils.make_variables(
        backend,
        Symbol("γ_$(player)_$(level)"),
        length(G_lower),
    )

    L = J - λ_level' * F_lower - γ_level' * G_lower
    ∇L = Symbolics.gradient(L, z_lower)

    F̃ = Vector{symbolic_type}([∇L; F_lower])
    G̃ = Vector{symbolic_type}([G_lower; γ_level; θ - γ_level' * G_lower])
    z̃ = [z_lower; λ_level; γ_level]

    return (; F = F̃, G = G̃, z = z̃)
end