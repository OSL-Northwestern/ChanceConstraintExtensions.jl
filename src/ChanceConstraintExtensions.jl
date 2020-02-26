module ChanceConstraintExtensions

using JuMP
import Statistics

export @chance_constraint

"""
    Γ(y::T, ε::Real)

Spline that runs from +1.0 to 0.0 over the domain `[-ε, ε]`.
"""
function Γ(y::T, ε::Real) where {T <: Real}
    if y <= -ε
        return one(T)
    elseif y >= ε
        return zero(T)
    else
        return 15 / 16 * (-1 / 5 * (y / ε)^5 + 2 / 3 * (y / ε)^3 - (y / ε) + 8 / 15)
    end
end

"""
    chance_constraint(
        model::Model,
        expr::Vector{NonlinearExpression};
        probability::Float64,
        smoothing::Float64 = 0.01,
        bisection_atol::Float64 = 1e-6,
        bisection_rtol::Float64 = 0.0,
    )

Add a chance constraint to `model`, where `P(expr <= 0.0) >= 1 - probability`.

- `smoothing` is the smoothing parameter `ϵ` in the smoothed quantile funciton.
- `bisection_atol` and `bisection_rtol` are used as the termination tolerance
  in the bisection search for the root of the quantile `y`.
"""
function chance_constraint(
    model::JuMP.Model,
    expr::Vector{JuMP.NonlinearExpression};
    probability::Float64,
    smoothing::Float64 = 0.01,
    bisection_atol::Float64 = 1e-10,
    bisection_rtol::Float64 = 0.0
)
    """
        g(z...)

    Solve for root `y` of `Σᵢ Γ(zᵢ - y) = (1-α)N`.
    """
    function g(z::T...) where {T <: Real}
        # Since we know that `y` is the quantile of `C(x,ω)`, we can derive good initial
        # bounds from the min and max of `C(x,ω)`.
        y_l, y_u = extrema(z)
        # We can also take a good guess at the inital root as the empirical quantile.
        # This should put us in the rough ballpark.
        # TODO(odow): we could potentially be clever and evaluate a couple of tigher bounds
        # for y_l and y_m, but evaluating at 1 - probability ± δ.
        # If we get +ve and -ve values great. Otherwise fallback to the extrema.
        y_m = Statistics.quantile(z, 1 - probability)
        while !isapprox(y_l, y_u; atol = bisection_atol, rtol = bisection_rtol)
            # Since we know that the quantile function is monotonic, we only need to keep
            # track of whether the current iterate is above or below the probability
            # threshold. We don't need to evaluate and compare the actual function values.
            if sum(Γ(zi - y_m, smoothing) for zi in z) > (1 - probability) * length(z)
                y_u = y_m
            else
                y_l = y_m
            end
            y_m = (y_l + y_u) / 2
        end
        return (y_l + y_u) / 2
    end
    # Some JuMP magic:
    # (a): create a new symbol that can't use used as an identifies in user-space
    g_sym = gensym()
    # (b): register the JuMP function and auto-diff it using ForwardDiff
    JuMP.register(model, g_sym, length(expr), g; autodiff = true)
    # (c): create the constraint `@NLconstraint(model, g(expr...) <= 0)`, but use the
    #      function syntax to avoid nasties in the macros.
    return JuMP.add_NL_constraint(model, :($(Expr(:call, g_sym, expr...)) <= 0))
end

function _is_n_call(expr::Expr, n::Int, sym::Symbol)
    return expr.head == :call && length(expr.args) == n + 1 && expr.args[1] == sym
end

_is_n_call(expr, n, sym) = false

function _parse_chance_expression(expr::Expr)
    # TODO(odow): re-write these sanity checks as informative error messages.
    @assert _is_n_call(expr, 2, :(>=))
    P_expr = expr.args[2]
    @assert _is_n_call(P_expr, 2, :P)
    lhs = if _is_n_call(P_expr.args[2], 2, :(<=))
        Expr(:call, :-, P_expr.args[2].args[2], P_expr.args[2].args[3])
    elseif _is_n_call(P_expr.args[2], 2, :(>=))
        Expr(:call, :-, P_expr.args[2].args[3], P_expr.args[2].args[2])
    else
        @assert false
    end
    @assert expr.args[2].args[3] isa Expr
    @assert expr.args[2].args[3].head == :kw
    ω = P_expr.args[3].args[1]
    Ω = P_expr.args[3].args[2]
    α = esc(Expr(:call, :-, 1, expr.args[3]))
    return lhs, ω, Ω, α
end

macro chance_constraint(model, expr, kwargs...)
    smoothing = 0.01
    atol = 1e-10
    rtol = 0.0
    for kwarg in kwargs
        if kwarg.args[1] == :smoothing
            smoothing = esc(kwarg.args[2])
        elseif kwarg.args[1] == :bisection_atol
            atol = esc(kwarg.args[2])
        elseif kwarg.args[1] == :bisection_rtol
            rtol = esc(kwarg.args[2])
        end
    end
    lhs, y, Y, α = _parse_chance_expression(expr)
    code = quote
        # nl_expr = @NLexpression(model, [ω = Ω], x^2 + ω - 2)
        # nl_expr = $(Expr(
        #     :macrocall,
        #     @__LINE__,
        #     Symbol("@NLexpression"),
        #     esc(model),
        #     :([ω = Ω]),
        #     :(x^2 + ω - 2)
        # ))
        # nl_expr = esc(@NLexpression(
        #     $(esc(model)),
        #     [$y = $(esc(Y))],
        #     $(esc(lhs))
        # ))
        nl_expr = NonlinearExpression[
            @NLexpression($(esc(model)), $(esc(lhs)))
            for $(esc(y)) in $(esc(Y))
        ]
        chance_constraint(
            $(esc(model)),
            collect(nl_expr);
            probability = $α,
            smoothing = $smoothing,
            bisection_atol = $atol,
            bisection_rtol = $rtol,
        )
    end
    @show code
    return code
end

end

# using JuMP
# model = Model()
# @variable(model, x)
# Ω = 1:2
# @chance_constraint(model, P(ω * x <= 1, ω = Ω) >= 0.9)
# @chance_constraint(model, P(ω * x <= 1, ω = Ω) >= 0.9, smoothing = 0.1)
# # @chance_constraint(model, P(ω * x <= 1, ω = Ω) >= 0.9, kwargs...)

