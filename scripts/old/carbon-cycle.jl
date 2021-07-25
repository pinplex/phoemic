# load packages
using DifferentialEquations
using Flux
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
#p2 = initial_params(NN)
p2, re = Flux.destructure(NN)
p3 = deepcopy(p2)
ps = Flux.params(p2)

#%%
function global_carbon_cycle_hybrid(du, u, p, t)
  T, Cl, Co, Ca = u
  #α, β, σ, η, tau_a, tau_l, Q10 = p[1:7] # parameters of process model
  #p = p[8:end] # parameters of neural netword
  CO2 = Ca * PgC_to_ppm
  #CO2 = abs(CO2)

  du[1] = dT = 1/tau_a*(α*log(CO2/CO2_ref) - (T - T_ref)) # temperature
  #du[2] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - re(p)([T - T_ref, Cl])[1] # land carbon
  du[2] = dCl = NN([CO2/CO2_ref,T/T_ref,Cl/Cl_ref],p)[1] # land carbon
  #du[2] = dCl = NPP_ref*(1 + re(p)([log(CO2/CO2_ref)])[1]) - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  #du[2] = dCl = NPP_ref*re(p)([CO2/CO2_ref])[1] - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  #du[2] = dCl = NPP_ref*re(p)([log(CO2/CO2_ref)])[1] - Cl/tau_l*Q10^((T - T_ref)/10) # land carbon
  #du[2] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/tau_l * re(p)([T - T_ref])[1] # land carbon
  du[3] = dCo = σ*((Ca - Ca_ref) - η*(Co - Co_ref)) # ocean carbon
  du[4] = dCa = - dCl - dCo # atmos carbon

end

# %% solve the hybrid problem
prob_hybrid = ODEProblem(global_carbon_cycle_hybrid, u0, tspan, p=p2)
sol_hybrid = solve(prob_hybrid, Tsit5(), p=p2, saveat=1)

#%% make plot
pl = plot(sol_true, layout=(2,2), label="true")
plot!(pl, sol_hybrid, layout=(2,2), label="pred")

#%% set up training
opt = ADAM(0.01)
t = range(tspan[1],tspan[2],length=len)

function predict_true()
  Array(solve(prob_true, Tsit5(), saveat=1))
end

ode_data = predict_true()

function predict_hybrid()
  Array(solve(prob_hybrid, Tsit5(), p=p2, saveat=1))
end

function loss_mass_balance()
    pdata = predict_hybrid()
    tdata = ode_data #predict_true()
    loss = sum(abs2, sum([pdata[2,:] .- pdata[2,1],
                          pdata[3,:] .- pdata[3,1],
                          tdata[4,:] .- tdata[4,1]]))
    #pland = pdata[2,:] .- pdata[2,1]
    #pocean = pdata[3,:] .- pdata[3,1]
    #loss = loss + sum(abs2, pland[pland.<0])
    loss
end

function loss_mse()
    pdata = predict_hybrid()
    tdata = ode_data #predict_true()
    loss = Flux.Losses.mse(tdata[2,:], pdata[2,:]) # ocean
    loss
end

cb = function (;doplot=false) #callback function to observe training
  pred = predict_hybrid()
  display(loss_mse())
  # plot current prediction against data
  pl = plot(t,ode_data', layout=(2,2), label="data",
            title=["Temp" "C_land" "C_ocean" "C_atmos"])
  plot!(pl,t,pred', layout=(2,2), label="prediction")
  display(plot(pl))
  return false
end

#%% train
data = Iterators.repeated((), 1000)
#Flux.train!(loss_mass_balance, ps, data, opt, cb = cb)
Flux.train!(loss_mse, ps, data, opt, cb = cb)

#%% make final plot
pred = predict_hybrid()
display(sum(abs2,ode_data .- pred))
# plot current prediction against data
pl = scatter(t,ode_data', layout=(2,2), label="data",
             title=["Temp" "C_land" "C_ocean" "C_atmos"])
scatter!(pl,t,pred', layout=(2,2), label="prediction")
savefig("no-Cland-constraint.png")

#%% pre-train
