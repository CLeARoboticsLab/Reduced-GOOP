"""
    PrimalCoordinateSpec

Problem-level semantics for one coordinate of a player's flattened primal
vector. `shift_rule` is one of `:successor`, `:identity`, `:reset`, or
`:unsupported`. A `:successor` coordinate at destination stage `t` is
transported from source stage `t + 1`; `tail_role` documents how the final
primal knot itself is completed. `horizon_role` and `exact_shift_invariant`
may be authored when the objective or constraint stencil makes the boundary
width impossible to infer from coordinate availability alone.
"""
Base.@kwdef struct PrimalCoordinateSpec
	player::Int
	variable::Symbol = :generic_primal
	stage::Union{Nothing, Int} = nothing
	component::Int
	shift_rule::Symbol = :unsupported
	tail_role::Symbol = :unspecified
	horizon_role::Symbol = :infer
	exact_shift_invariant::Union{Nothing, Bool} = nothing
end

"""
    EqualityCoordinateSpec

Problem-level semantics for one scalar equality constraint. `stage` identifies
the horizon stage of the equation, while `successor_stage` can identify the
state knot on the right side of a transition equation (for example, dynamics
at stage `t` has `successor_stage == t + 1`). As with primal coordinates,
problem authors may override inferred horizon localization through
`horizon_role` and `exact_shift_invariant`.
"""
Base.@kwdef struct EqualityCoordinateSpec
	scope::Symbol = :player
	player::Union{Nothing, Int} = nothing
	equation_class::Symbol = :unclassified
	equation_type::Symbol = :generic_equality
	stage::Union{Nothing, Int} = nothing
	successor_stage::Union{Nothing, Int} = nothing
	component::Int
	shift_rule::Symbol = :unsupported
	tail_role::Symbol = :unspecified
	horizon_role::Symbol = :infer
	exact_shift_invariant::Union{Nothing, Bool} = nothing
end

"""
    GOOPSemanticLayout

Explicit, problem-authored coordinate semantics. The vectors must have exactly
the same lengths and ordering as the corresponding flattened primal and
equality functions. The KKT generator validates this contract before using it.
"""
Base.@kwdef struct GOOPSemanticLayout
	primal_by_player::Vector{Vector{PrimalCoordinateSpec}}
	equality_by_player::Vector{Vector{EqualityCoordinateSpec}}
	shared_equality::Vector{EqualityCoordinateSpec} = EqualityCoordinateSpec[]
end

"""
Metadata for one semantically meaningful generated KKT variable coordinate.

For `family == :stationarity_multiplier`, `owner_level` is the hierarchy level
whose reduced Lagrangian owns the multiplier and `target_level` is the lower
stationarity equation multiplied by that coordinate. Thus the mathematical
`ψ_in` block is selected by `target_level == num_levels[player]`.
`successor_exists` records whether a same-semantic stage-`t+1` multiplier is
available as the one-step source; `tail_role` records the terminal-completion
semantics even when the selected transport policy resets or holds the tail.
"""
Base.@kwdef struct KKTVariableCoordinate
	index::Int
	family::Symbol
	scope::Symbol
	player::Union{Nothing, Int} = nothing
	owner_level::Union{Nothing, Int} = nothing
	target_level::Union{Nothing, Int} = nothing
	equation_class::Symbol = :not_applicable
	equation_type::Symbol = :none
	primal_variable::Union{Nothing, Symbol} = nothing
	stage::Union{Nothing, Int} = nothing
	successor_stage::Union{Nothing, Int} = nothing
	component::Int
	shift_rule::Symbol = :unsupported
	successor_exists::Bool = false
	tail_role::Symbol = :unspecified
end

"""
Metadata for one generated KKT residual row.

`horizon_role` localizes the row as one of `:inherited_interior`,
`:initial_boundary`, `:terminal_boundary`, or `:global`.  The Boolean
`exact_shift_invariant` is deliberately stricter than merely having a stage:
it is true only when the row should be inherited unchanged under an exact
one-stage receding-horizon shift (before accounting for changed problem
data).  Boundary and global rows therefore remain explicit rather than being
silently pooled with the inherited interior.
"""
Base.@kwdef struct KKTEquationCoordinate
	row::Int
	family::Symbol
	scope::Symbol
	player::Union{Nothing, Int} = nothing
	level::Union{Nothing, Int} = nothing
	equation_class::Symbol = :not_applicable
	equation_type::Symbol
	primal_variable::Union{Nothing, Symbol} = nothing
	stage::Union{Nothing, Int} = nothing
	successor_stage::Union{Nothing, Int} = nothing
	component::Int
	horizon_role::Symbol = :unclassified
	exact_shift_invariant::Bool = false
end

"""
Semantic metadata emitted alongside a generated KKT system.

`variables` intentionally contains the primal, equality-multiplier, and
hierarchical-stationarity-multiplier coordinates needed for selective warm
starts. Other slack and complementarity variables retain the existing coarse
coordinate fields on `GOOPKKTSystem`.
"""
Base.@kwdef struct GOOPKKTMetadata
	variables::Vector{KKTVariableCoordinate}
	equations::Vector{KKTEquationCoordinate}
end

"""A deterministic source-to-destination receding-horizon dual permutation."""
Base.@kwdef struct KKTDualTransportMap
	source_indices::Vector{Int}
	destination_indices::Vector{Int}
	reset_indices::Vector{Int}
	tail::Symbol
end

const _SEMANTIC_SHIFT_RULES = (:successor, :identity, :reset, :unsupported)
const _DUAL_TRANSPORT_TAILS = (:zero, :hold)
const _KKT_EQUATION_HORIZON_ROLES = (
	:inherited_interior,
	:initial_boundary,
	:terminal_boundary,
	:global,
)
const _SEMANTIC_SPEC_HORIZON_ROLES = (:infer, _KKT_EQUATION_HORIZON_ROLES...)
const _EQUALITY_EQUATION_CLASSES = (
	:initial_condition,
	:dynamics,
	:time_indexed_path_equality,
	:shared_time_indexed,
	:global_non_time_indexed,
)

function _validate_equation_class(spec::EqualityCoordinateSpec, label)
	class = spec.equation_class
	class === :unclassified && return
	class in _EQUALITY_EQUATION_CLASSES || throw(
		ArgumentError(
			"Unknown equality equation class $(class) for $(label); expected one " *
			"of $(collect(_EQUALITY_EQUATION_CLASSES)).",
		),
	)
	if class in (:initial_condition, :dynamics, :time_indexed_path_equality)
		spec.scope === :player || throw(
			ArgumentError("$(class) requires scope=:player for $(label)."),
		)
		isnothing(spec.player) && throw(
			ArgumentError("$(class) requires a player for $(label)."),
		)
		isnothing(spec.stage) && throw(
			ArgumentError("$(class) requires an explicit stage for $(label)."),
		)
	elseif class === :shared_time_indexed
		spec.scope === :shared || throw(
			ArgumentError(":shared_time_indexed requires scope=:shared for $(label)."),
		)
		isnothing(spec.player) || throw(
			ArgumentError(":shared_time_indexed requires player=nothing for $(label)."),
		)
		isnothing(spec.stage) && throw(
			ArgumentError(":shared_time_indexed requires an explicit stage for $(label)."),
		)
	else
		@assert class === :global_non_time_indexed
		isnothing(spec.stage) || throw(
			ArgumentError(":global_non_time_indexed requires stage=nothing for $(label)."),
		)
	end
end

function _validate_shift_rule(rule::Symbol, label)
	rule in _SEMANTIC_SHIFT_RULES || throw(
		ArgumentError(
			"Unknown semantic shift rule $(rule) for $(label); expected one of " *
			"$(collect(_SEMANTIC_SHIFT_RULES)).",
		),
	)
end

function _validate_horizon_annotation(spec, label)
	role = spec.horizon_role
	role in _SEMANTIC_SPEC_HORIZON_ROLES || throw(
		ArgumentError(
			"Unknown horizon role $(role) for $(label); expected one of " *
			"$(collect(_SEMANTIC_SPEC_HORIZON_ROLES)).",
		),
	)
	if role !== :infer && !isnothing(spec.exact_shift_invariant)
		spec.exact_shift_invariant == (role === :inherited_interior) || throw(
			ArgumentError(
				"Inconsistent exact-shift annotation for $(label): " *
				"horizon_role=$(role), exact_shift_invariant=" *
				"$(spec.exact_shift_invariant).",
			),
		)
	end
	nothing
end

function _validate_semantic_layout(
	layout::GOOPSemanticLayout,
	primal_dims,
	equality_dims,
	shared_equality_dims,
	num_players,
)
	length(layout.primal_by_player) == num_players || throw(
		DimensionMismatch(
			"Semantic layout has $(length(layout.primal_by_player)) primal player " *
			"blocks; expected $(num_players).",
		),
	)
	length(layout.equality_by_player) == num_players || throw(
		DimensionMismatch(
			"Semantic layout has $(length(layout.equality_by_player)) equality player " *
			"blocks; expected $(num_players).",
		),
	)

	for player in 1:num_players
		primal = layout.primal_by_player[player]
		equality = layout.equality_by_player[player]
		length(primal) == primal_dims[player] || throw(
			DimensionMismatch(
				"Player $(player) semantic primal layout has length $(length(primal)); " *
				"expected $(primal_dims[player]).",
			),
		)
		length(equality) == equality_dims[player] || throw(
			DimensionMismatch(
				"Player $(player) semantic equality layout has length " *
				"$(length(equality)); expected $(equality_dims[player]).",
			),
		)

		for (component, spec) in enumerate(primal)
			spec.player == player || throw(
				ArgumentError(
					"Primal semantic coordinate $(component) in player block $(player) " *
					"claims player $(spec.player).",
				),
			)
			spec.component >= 1 || throw(
				ArgumentError("Primal semantic components must be positive."),
			)
			_validate_shift_rule(spec.shift_rule, "player $(player) primal coordinate")
			_validate_horizon_annotation(
				spec,
				"player $(player) primal coordinate $(component)",
			)
			spec.shift_rule === :successor && isnothing(spec.stage) && throw(
				ArgumentError(
					"A :successor primal coordinate requires an explicit stage " *
					"(player $(player), coordinate $(component)).",
				),
			)
		end

		for (component, spec) in enumerate(equality)
			spec.scope === :player || throw(
				ArgumentError(
					"Player equality semantic coordinate $(component) must use " *
					"scope=:player, got $(spec.scope).",
				),
			)
			spec.player == player || throw(
				ArgumentError(
					"Equality semantic coordinate $(component) in player block " *
					"$(player) claims player $(spec.player).",
				),
			)
			spec.component >= 1 || throw(
				ArgumentError("Equality semantic components must be positive."),
			)
			_validate_equation_class(
				spec,
				"player $(player) equality coordinate $(component)",
			)
			_validate_shift_rule(spec.shift_rule, "player $(player) equality coordinate")
			_validate_horizon_annotation(
				spec,
				"player $(player) equality coordinate $(component)",
			)
			spec.shift_rule === :successor && isnothing(spec.stage) && throw(
				ArgumentError(
					"A :successor equality coordinate requires an explicit stage " *
					"(player $(player), coordinate $(component)).",
				),
			)
		end
	end

	length(layout.shared_equality) == shared_equality_dims || throw(
		DimensionMismatch(
			"Shared semantic equality layout has length " *
			"$(length(layout.shared_equality)); expected $(shared_equality_dims).",
		),
	)
	for (component, spec) in enumerate(layout.shared_equality)
		spec.scope === :shared || throw(
			ArgumentError(
				"Shared equality semantic coordinate $(component) must use " *
				"scope=:shared, got $(spec.scope).",
			),
		)
		isnothing(spec.player) || throw(
			ArgumentError("Shared equality semantic coordinates must use player=nothing."),
		)
		spec.component >= 1 ||
			throw(ArgumentError("Shared equality semantic components must be positive."))
		_validate_equation_class(spec, "shared equality coordinate $(component)")
		_validate_shift_rule(spec.shift_rule, "shared equality coordinate")
		_validate_horizon_annotation(spec, "shared equality coordinate $(component)")
		spec.shift_rule === :successor && isnothing(spec.stage) && throw(
			ArgumentError(
				"A :successor shared equality coordinate requires an explicit stage " *
				"(coordinate $(component)).",
			),
		)
	end
	layout
end

function _generic_semantic_layout(
	primal_dims,
	equality_dims,
	shared_equality_dims,
	num_players,
)
	primal_by_player = [
		[
			PrimalCoordinateSpec(;
				player,
				variable = :generic_primal,
				component,
				shift_rule = :unsupported,
			) for component in 1:primal_dims[player]
		] for player in 1:num_players
	]
	equality_by_player = [
		[
			EqualityCoordinateSpec(;
				scope = :player,
				player,
				equation_type = :generic_equality,
				component,
				shift_rule = :unsupported,
			) for component in 1:equality_dims[player]
		] for player in 1:num_players
	]
	shared_equality = [
		EqualityCoordinateSpec(;
			scope = :shared,
			player = nothing,
			equation_type = :generic_shared_equality,
			component,
			shift_rule = :unsupported,
		) for component in 1:shared_equality_dims
	]
	GOOPSemanticLayout(; primal_by_player, equality_by_player, shared_equality)
end

function _resolved_semantic_layout(
	layout,
	primal_dims,
	equality_dims,
	shared_equality_dims,
	num_players,
)
	resolved = isnothing(layout) ?
			   _generic_semantic_layout(
		primal_dims,
		equality_dims,
		shared_equality_dims,
		num_players,
	) :
			   layout
	resolved isa GOOPSemanticLayout || throw(
		ArgumentError(
			"semantic_layout must be a GOOPSemanticLayout or nothing, got " *
			"$(typeof(resolved)).",
		),
	)
	_validate_semantic_layout(
		resolved,
		primal_dims,
		equality_dims,
		shared_equality_dims,
		num_players,
	)
end

function _require_kkt_metadata(kkt)
	hasproperty(kkt, :metadata) || throw(
		ArgumentError("This KKT object has no semantic metadata field."),
	)
	metadata = getproperty(kkt, :metadata)
	metadata isa GOOPKKTMetadata || throw(
		ArgumentError(
			"This KKT system has no generated semantic metadata; regenerate the " *
			"reduced or quasi KKT system from an annotated ParametricGOOP.",
		),
	)
	metadata
end

"""Return a fresh vector of generated KKT variable-coordinate metadata."""
function kkt_variable_metadata(kkt; family::Union{Nothing, Symbol} = nothing)
	metadata = _require_kkt_metadata(kkt)
	records = isnothing(family) ?
			  metadata.variables :
			  filter(record -> record.family === family, metadata.variables)
	copy(records)
end

"""Return a fresh vector of generated KKT residual-row metadata."""
function kkt_equation_metadata(kkt; family::Union{Nothing, Symbol} = nothing)
	metadata = _require_kkt_metadata(kkt)
	records = isnothing(family) ?
			  metadata.equations :
			  filter(record -> record.family === family, metadata.equations)
	copy(records)
end

function _dual_transport_key(record::KKTVariableCoordinate)
	(
		record.family,
		record.scope,
		record.player,
		record.owner_level,
		record.target_level,
		record.equation_class,
		record.equation_type,
		record.primal_variable,
		record.component,
	)
end

"""
    receding_dual_transport_map(kkt; tail = :zero)

Construct the semantic one-stage source-to-destination map for equality and
hierarchical-stationarity multipliers. `tail=:zero` resets a `:successor`
coordinate with no stage-`t+1` source; `tail=:hold` retains its same-stage
source. Explicit `:reset` coordinates are reset under both tail policies.
"""
function receding_dual_transport_map(kkt; tail::Symbol = :zero)
	tail in _DUAL_TRANSPORT_TAILS || throw(
		ArgumentError(
			"Unknown dual transport tail policy $(tail); expected :zero or :hold.",
		),
	)
	metadata = _require_kkt_metadata(kkt)
	duals = sort!(
		filter(
			record ->
				record.family in (:equality_multiplier, :stationarity_multiplier),
			metadata.variables,
		);
		by = record -> record.index,
	)

	lookup = Dict{Tuple, KKTVariableCoordinate}()
	for record in duals
		key = (_dual_transport_key(record)..., record.stage)
		haskey(lookup, key) && throw(
			ArgumentError(
				"Semantic dual coordinates are not unique for transport key $(key).",
			),
		)
		lookup[key] = record
	end

	source_indices = Int[]
	destination_indices = Int[]
	reset_indices = Int[]
	unsupported = Int[]
	for destination in duals
		rule = destination.shift_rule
		if rule === :identity
			push!(source_indices, destination.index)
			push!(destination_indices, destination.index)
		elseif rule === :reset
			push!(reset_indices, destination.index)
		elseif rule === :unsupported
			push!(unsupported, destination.index)
		else
			@assert rule === :successor
			@assert !isnothing(destination.stage)
			source_key =
				(_dual_transport_key(destination)..., destination.stage + 1)
			source_available = haskey(lookup, source_key)
			source_available == destination.successor_exists || throw(
				ArgumentError(
					"Multiplier coordinate $(destination.index) declares " *
					"successor_exists=$(destination.successor_exists), but generated " *
					"transport lookup found successor=$(source_available).",
				),
			)
			if source_available
				push!(source_indices, lookup[source_key].index)
				push!(destination_indices, destination.index)
			elseif tail === :hold
				push!(source_indices, destination.index)
				push!(destination_indices, destination.index)
			else
				push!(reset_indices, destination.index)
			end
		end
	end

	isempty(unsupported) || throw(
		ArgumentError(
			"Receding dual transport requires explicit semantic shift rules; " *
			"unsupported multiplier coordinates begin at " *
			"$(unsupported[1:min(end, 8)]).",
		),
	)
	KKTDualTransportMap(;
		source_indices,
		destination_indices,
		reset_indices,
		tail,
	)
end

"""
    transport_receding_duals(previous_z, kkt; tail = :zero)

Return a fresh full KKT vector whose equality/stationarity multipliers have
been transported by `receding_dual_transport_map`. Non-dual coordinates are
left unchanged so the result can be passed as the dual source to
`build_selective_warmstart`.
"""
function transport_receding_duals(previous_z, kkt; tail::Symbol = :zero)
	length(previous_z) == kkt.variable_dimension || throw(
		DimensionMismatch(
			"Previous KKT point has length $(length(previous_z)); expected " *
			"$(kkt.variable_dimension).",
		),
	)
	map = receding_dual_transport_map(kkt; tail)
	transported = copy(previous_z)
	transported[map.reset_indices] .= zero(eltype(transported))
	transported[map.destination_indices] .= previous_z[map.source_indices]
	transported
end
