using ChanceConstraintExtensions
import Ipopt
using JuMP
using Test

@testset "Γ" begin
    @test ChanceConstraintExtensions.Γ(-1.1, 1.0) == 1.0
    @test ChanceConstraintExtensions.Γ(-0.11, 0.1) == 1.0
    @test ChanceConstraintExtensions.Γ(0.11, 0.1) == 0.0
    @test ChanceConstraintExtensions.Γ(1.1, 1.0) == 0.0
    @test ChanceConstraintExtensions.Γ(0.0, 0.1) ≈ 0.5
    @test ChanceConstraintExtensions.Γ(0.0, 0.2) ≈ 0.5
    @test ChanceConstraintExtensions.Γ(0.05, 0.1) ≈ 53 / 512
    @test ChanceConstraintExtensions.Γ(-0.05, 0.1) ≈ 1 - 53 / 512
end

@testset "Example" begin
    Ω = [-1.0, -0.2, 0.2, 1.0]
    K = length(Ω)
    model = Model(with_optimizer(Ipopt.Optimizer))
    set_parameter(model, "print_level", 0)
    @variable(model, -2 <= x <= 2, start = 0.4)
    chance_constraint(
        model,
        @NLexpression(model, [k = 1:K], x^2 - 2 + Ω[k]),
        probability = 0.3, smoothing = 0.01, bisection_atol=1e-4
    )
    @objective(model, Max, x)
    optimize!(model)
    @test value(x) ≈ 1.34035 atol = 1e-4
end
