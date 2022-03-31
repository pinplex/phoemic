using Pkg
Pkg.activate(".")
#using Pkg; Pkg.activate("."); Pkg.instantiate()

using Flux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
println("Starting Soon!")
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
using OrdinaryDiffEq
#import ModelingToolkit: Interval, infimum, supremum
using DomainSets
using Random
using Plots
using DifferentialEquations

##
Random.seed!(1234)

ex_input(t) = @. 10.0 * sin(2.0 * pi * t)
@register ex_input(t)

## This the the PDE definition code
@parameters t ,σ_ ,β, ρ 
@variables x(..), y(..), z(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_*(y(t) - x(t)),
       Dt(y(t)) ~ x(t)*(ρ - z(t)) - y(t),
       Dt(z(t)) ~ x(t)*y(t) - β*z(t) - ex_input(t)]


bcs = [x(0) ~ -8.0, y(0) ~ 7.0, z(0) ~ 27.0]
domains = [t ∈ Interval(0.0,1.0)]
dt = 0.05

input_ = length(domains)
n = 12
chain = [FastChain(FastDense(input_,n,Flux.tanh),FastDense(n,n,Flux.σ),FastDense(n,1)) for _ in 1:3]

## Generate Data
function lorenz!(du,u,p,t)
    du[1] = 10.0*(u[2]-u[1])
    du[2] = u[1]*(28.0-u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3)*u[3] - ex_input(t)
end

u0 = [-8.0; 7.0; 27.0]
tspan = (0.0,1.0)

prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob, Tsit5(), dt=0.1)
ts = [infimum(d.domain):dt:supremum(d.domain) for d in domains][1]

plot(sol)

function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us,ts_]
end

data = getData(sol)

## Additional Loss Function
initθs = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))
acum =  [0;accumulate(+, length.(initθs))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
(u_ , t_) = data
len = length(data[2])

function additional_loss(phi, θ , p)
    return sum(sum(abs2, phi[i](t_ , θ[sep[i]]) .- u_[[i], :])/len for i in 1:1:3)
end

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             NeuralPDE.GridTraining(0.05);
                                             #QuadratureTraining();
                                             init_params =initθs,
                                             param_estim=true,
                                             additional_loss=additional_loss)
testθ =reduce(vcat,initθs)
additional_loss(discretization.phi, testθ, nothing)

@named pde_system = PDESystem(eqs,bcs,domains,
                      [t],[x(t), y(t), z(t)],[σ_, ρ, β],
                      defaults=Dict([p => 1.0 for p in [σ_, ρ, β]]))
                      
                      pde_system.ps
                      pde_system.defaults
                      prob

prob = NeuralPDE.discretize(pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(pde_system,discretization)
prob.f.f.loss_function([testθ;ones(3)])

@time res = GalacticOptim.solve(prob, Optim.BFGS(); maxiters=6000)
prob = NeuralPDE.discretize(pde_system, discretization)
prob.f.f.loss_function([testθ;res. minimizer[end-2:end]])
additional_loss(discretization.phi, res.minimizer[1:end], nothing)

p_ = res.minimizer[end-2:end]
@test sum(abs2, p_[1] - 10.00) < 0.1
@test sum(abs2, p_[2] - 28.00) < 0.1
@test sum(abs2, p_[3] - (8/3)) < 0.1

## Plotting the system
#discretization.default_p  # = [1.0,1.0,1.0]
initθ = discretization.init_params
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]
ts = [infimum(d.domain):dt/10:supremum(d.domain) for d in domains][1]
u_predict  = [[discretization.phi[i]([t],minimizers[i])[1] for t in ts] for i in 1:3]
plot(sol)
plot!(ts, u_predict, label = ["x(t)" "y(t)" "z(t)"])


# TODO generalization with other parameters?