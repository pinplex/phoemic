# load packages
using DifferentialEquations
using Flux
using Plots
using DiffEqFlux
using Optim
using DiffEqSensitivity

#%% parameters
# conversion factors
PgC_to_ppm = 0.471 # ppm / PgC

# initial values
T_ref = 288 # K
CO2_ref = 280 # ppm
Ca_ref = CO2_ref / PgC_to_ppm
Cl_ref = 2500 # PgC
Co_ref = 37000 # PgC
NPP_ref = 60 # PgC / yr

# model parameters
α = 3.5 # K / ln(2); climate sensitivity
β = 0.4 # CO2 fertilization parameterization
σ = 0.015 # parameter for Henry's law
η = 10 * Ca_ref / Co_ref # ocean C uptake saturation
tau_a = 30 # yr, atmosphere residence time
tau_l = 41 # yr, land residence time
Q10 = 1.8

#%% define conventional model
function global_carbon_cycle(du, u, p, t)
  T, Cl, Co, Ca = u
  α, β, σ, η, tau_a, tau_l, Q10 = p
  CO2 = Ca * PgC_to_ppm

  du[1] = dT = 1/tau_a*(α*log(CO2/CO2_ref) - (T - T_ref)) # temperature
  du[2] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  du[3] = dCo = σ*((Ca - Ca_ref) - η*(Co - Co_ref)) # ocean carbon
  du[4] = dCa = - dCl - dCo # atmos carbon

end

# %% solve the problem
# initial conditions
u0 = [T_ref, Cl_ref, Co_ref, Ca_ref*2] # T, Cl, Co, Ca
len = 250
tspan = (1.0,Float64(len))
p1 = [α, β, σ, η, tau_a, tau_l, Q10]

prob_true = ODEProblem(global_carbon_cycle, u0, tspan, p1)
sol_true = solve(prob_true, Tsit5(), saveat=1)

#%% define neural model
# set up NN
NN = FastChain(FastDense(3, 32, tanh), FastDense(32, 32, tanh), FastDense(32,1))
p_random = Float64.(initial_params(NN))
p = p_random
NN([1,2,3],p_random)

#%%
function global_carbon_cycle_hybrid(du, u, p, t)
  T, Cl, Co, Ca = u
  #α, β, σ, η, tau_a, tau_l, Q10 = p[1:7] # parameters of process model
  #p = p[8:end] # parameters of neural netword
  CO2 = Ca * PgC_to_ppm
  #CO2 = abs(CO2)

  du[1] = dT = 1/tau_a*(α*log(CO2/CO2_ref) - (T - T_ref)) # temperature
  #du[2] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - re(p)([T - T_ref, Cl])[1] # land carbon
  du[2] = dCl = NN([CO2/CO2_ref,T/T_ref,Cl/Cl_ref],p)[1]# land carbon
  #du[2] = dCl = NPP_ref*(1 + re(p)([log(CO2/CO2_ref)])[1]) - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  #du[2] = dCl = NPP_ref*re(p)([CO2/CO2_ref])[1] - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  #du[2] = dCl = NPP_ref*re(p)([log(CO2/CO2_ref)])[1] - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  #du[2] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/tau_l * re(p)([T - T_ref])[1] # land carbon
  du[3] = dCo = σ*((Ca - Ca_ref) - η*(Co - Co_ref)) # ocean carbon
  du[4] = dCa = -dCl - dCo # atmos carbon

end

#%% solve the hybrid problem
prob_hybrid = ODEProblem(global_carbon_cycle_hybrid, u0, tspan, p)
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))
sol_hybrid = solve(prob_hybrid, TRBDF2(autodiff=false), p=p_random, saveat=1, sensealg = sensealg)

#%% make plot
pl = plot(sol_true, layout=(2,2), label="true", w=3,
          ylabel=["K" "Pg C" "Pg C" "Pg C"],
          title=["Temp" "C_land" "C_ocean" "C_atmos"])
plot!(pl, sol_hybrid, layout=(2,2), label="pred", w=3,
          title=["Temp" "C_land" "C_ocean" "C_atmos"])

savefig("Case02_learn-R_init.pdf")
#%% set up training
#opt = ADAM(0.1)
opt = BFGS(initial_stepnorm = 0.1)
t = range(tspan[1],tspan[2],length=len)

function predict_true()
  Array(solve(prob_true, Tsit5(), saveat=1))
end

ode_data = predict_true()

function predict_hybrid(p)
  Array(solve(prob_hybrid, Tsit5(), p=p, saveat=t, sensealg = sensealg))
end

function predict(θ)
  temp_hybridprob = remake(prob_hybrid;p=θ,saveat=1)
  Array(solve(temp_hybridprob, RK4();reltol=1e-8,abstol=1e-8,sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

function loss_mse(p)
    pdata = predict(p)
    tdata = ode_data #predict_true()
    loss = Flux.mse(pdata[2,:],tdata[2,:]) # ocean
    loss
end

cb = function (;doplot=false) #callback function to observe training
  pred = predict_hybrid(ps)
  display(loss_mse(ps))

  # plot current prediction against data
  pl = plot(true_data', layout=(2,2), label="data",  w=3)
  plot!(pred_data', layout=(2,2), label="prediction",  w=3,
        ylabel=["K" "Pg C" "Pg C" "Pg C"],
        title=["Temp" "C_land" "C_ocean" "C_atmos"])
  display(plot(pl))

  return false
end

#%% train
#data = Iterators.repeated((), 1000)
#Flux.train!(loss_mass_balance, ps, data, opt, cb = cb)
#Flux.train!(loss_mse, ps, data, opt)

losses = []
callback(θ,l) = begin
  push!(losses, l)
  if length(losses)%1==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  false
end
loss_mse(p_random)

#%%
oopz = DiffEqFlux.sciml_train(loss_mse, p_random, opt,
                              maxiters = 30, cb = callback,
                              allow_f_increases = true)

#%% make final plot
pred = predict_hybrid(oopz.minimizer)
display(sum(abs2,ode_data .- pred))
# plot current prediction against data
# pl = scatter(t,ode_data', layout=(2,2), label="data",
             # title=["Temp" "C_land" "C_ocean" "C_atmos"])
# scatter!(pl,t,pred', layout=(2,2), label="prediction")

#%%plot current prediction against data
pl = plot(ode_data', layout=(2,2), label="data",  w=3)
plot!(pred', layout=(2,2), label="prediction",  w=3,
      ylabel=["K" "Pg C" "Pg C" "Pg C"],
      title=["Temp" "C_land" "C_ocean" "C_atmos"])
display(plot(pl))
savefig("Case02_learn-R_after80iter.pdf")

#%% pre-train
