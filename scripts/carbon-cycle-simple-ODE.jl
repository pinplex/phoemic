# load packages
using DifferentialEquations
using Flux
using Flux.Data: DataLoader
using Plots
using DiffEqFlux

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

#%% define mechanistic model
function global_carbon_cycle(du, u, p, t)
  T, Cl, Co, Ca = u
  α, β, σ, η, tau_a, tau_l, Q10 = p
  CO2 = Ca * PgC_to_ppm

  du[1] = dT = 1/tau_a*(α*log(CO2/CO2_ref) - (T - T_ref)) # temperature
  du[2] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  du[3] = dCo = σ*((Ca - Ca_ref) - η*(Co - Co_ref)) # ocean carbon
  du[4] = dCa = - dCl - dCo # atmos carbon

end

#%% define one-param-to-learn model
function global_carbon_cycle_hybrid(du, u, p, t)
  T, Cl, Co, Ca = u
  β, Q10 = p # parameters to be learnt
  CO2 = Ca * PgC_to_ppm

  du[1] = dT = 1/tau_a*(α*log(CO2/CO2_ref) - (T - T_ref)) # temperature
  du[2] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  du[3] = dCo = σ*((Ca - Ca_ref) - η*(Co - Co_ref)) # ocean carbon
  du[4] = dCa = - dCl - dCo # atmos carbon

end

# %% solve the problem
# initial conditions: + 10 K
u0 = [T_ref+10, Cl_ref, Co_ref, Ca_ref] # T, Cl, Co, Ca:
len = 500
tspan = (1.0,Float64(len))
p = [α, β, σ, η, tau_a, tau_l, Q10]

# run solver
prob = ODEProblem(global_carbon_cycle, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=1)

#%% make plot
pl = plot(sol, layout=(2,2), label="sol",
          ylabel=["K" "Pg C" "Pg C" "Pg C"],
          title=["Temp" "C_land" "C_ocean" "C_atmos"])

#savefig("Case00_plus-10K.pdf")

# %% solve the problem
# initial conditions: impulse of 4 times atmospheric CO2
u0 = [T_ref, Cl_ref, Co_ref, Ca_ref*4] # T, Cl, Co, Ca:
len = 250
tspan = (1.0,Float64(len))
p = [α, β, σ, η, tau_a, tau_l, Q10]

# run solver
prob = ODEProblem(global_carbon_cycle, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=1)

#%% make plot
pl = plot(sol, layout=(2,2), label="sol",
          ylabel=["K" "Pg C" "Pg C" "Pg C"],
          title=["Temp" "C_land" "C_ocean" "C_atmos"])

#savefig("Case00_4xCO2.pdf")

#%% hybrid problem
p1 = [1.0,1.0] # initial guess of CO2 fertilization and Q10
prob_hybrid = ODEProblem(global_carbon_cycle_hybrid, u0, tspan, p1)
sol_hybrid = solve(prob_hybrid, Tsit5(), saveat=1)

#%% make plot
pred = Array(sol_hybrid)
true_data = Array(sol)

# plot current prediction against data
pl = plot(true_data', layout=(2,2), label="data",  w=3)
plot!(pred', layout=(2,2), label="prediction",  w=3,
      ylabel=["K" "Pg C" "Pg C" "Pg C"],
      title=["Temp" "C_land" "C_ocean" "C_atmos"])
display(plot(pl))
#savefig("Case01_learn-beta+Q10_init.pdf")

#%% set up training
lr = 0.1 # lower lr in the end
opt = ADAM(lr)
t = range(tspan[1],tspan[2],length=len)

#%%
function predict() # Our 1-layer "neural network"
    Array(solve(prob_hybrid, Tsit5(), p=p1, saveat=1))
end

# function loss_mass_balance()
#     pdata = predict()
#     tdata = Array(sol)
#     loss = sum(abs2, sum([pdata[2,:] .- pdata[2,1],
#                           pdata[3,:] .- pdata[3,1],
#                           tdata[4,:] .- tdata[4,1]]))
#     loss
# end

function loss_mse()
    pdata = predict()
    tdata = Array(sol)
    loss = Flux.Losses.mse(tdata[4,:], pdata[4,:]) # loss based on atmosphere CO2
    loss
end

cb = function (;doplot=false) #callback function to observe training
  pred_data = predict()
  true_data = Array(sol)
  display(loss_mse())

  # plot current prediction against data
  pl = plot(true_data', layout=(2,2), label="data",  w=3)
  plot!(pred_data', layout=(2,2), label="prediction",  w=3,
        ylabel=["K" "Pg C" "Pg C" "Pg C"],
        title=["Temp" "C_land" "C_ocean" "C_atmos"])
  display(plot(pl))
  return false
end

#%% train
ps = Flux.params(p1)
data = Iterators.repeated((), 100)
#Flux.train!(loss_mass_balance, ps, data, opt, cb = cb)
Flux.train!(loss_mse, ps, data, opt, cb = cb)

#%% make final plot
pred = predict()
true_data = Array(sol)

# plot current prediction against data
pl = plot(true_data', layout=(2,2), label="data",  w=3)
plot!(pred', layout=(2,2), label="prediction",  w=3,
      ylabel=["K" "Pg C" "Pg C" "Pg C"],
      title=["Temp" "C_land" "C_ocean" "C_atmos"])
display(plot(pl))
savefig("Case01_learn-beta+Q10.pdf")
