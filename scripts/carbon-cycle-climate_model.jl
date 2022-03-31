# load packages
using DifferentialEquations
using Flux
using Flux.Data: DataLoader
using Plots
using LaTeXStrings
using DiffEqFlux
using Interpolations
using ForwardDiff
using Optim
using DiffEqSensitivity
using CSV
using DataFrames

#%% define global variables
home = pwd()

#%% parameters
# constants
dt = 60. * 60. * 24. * 365. # one year expressed in seconds

# conversion factors
PgC_to_ppm = 0.471 # ppm / PgC

# initial values climate
T_ref = 288 # K
To_ref = 0.036825 # K
c_w = 4e3  #  Specific heat of fluid (water) in J/kg/K
ρ_w = 1e3  #  Density of water in kg/m3
H = 250 #  Depth of fluid (water) in m
C = c_w * ρ_w * H / dt # atmosphere/land/upper-ocean heat capacity, W yr-1 m-2 K-1

# partially overwrite values using mulit-model means by CMIP5; see Geoffroy et al. (2013)
C = 7.3 # atmosphere/land/upper-ocean heat capacity, W yr-1 m-2 K-1
Co = 91 # deep-ocean heat capacity, W yr-1 m-2 K-1
γ = 0.73 # heat exchange coefficient

# inital values carbon
CO2_ref = 280 # ppm
Ca_ref = CO2_ref / PgC_to_ppm
Cl_ref = 2470 # PgC
Co_ref = 37030 # PgC
NPP_ref = 60 # PgC / yr

# model parameters
α = 5.35 # empirical constant for radiative forcing
β = 0.4 # CO2 fertilization parameterization
σ = 0.015 # parameter for Henry's law
η = 10 * Ca_ref / Co_ref # ocean C uptake saturation
λ = 0.85 # climate feedback parameter
τ = 41 # yr, land residence time
Q10 = 1.8

#%% initial conditions:
u0 = [0, 0, 0, 0, T_ref, Cl_ref, Co_ref, Ca_ref] # Hp, Ho, H, T, To, Cl, Co, Ca
len = 140
tspan = (1.0,Float64(len))

#%% forcing data from LR_1pctCO2
#= n = 140
conc = 280 # ppm
k = Array{Float64}(undef, n) #or Float64 or Any instead of Int64
for i in 1:n
    conc = conc * 1.01
    k[i] = conc 
end =#
#%% read data
xs = range(1, length=len)
df = DataFrame(CSV.File(home*"/data/carbon-budget_MPI-ESM1-2-LR_1pctCO2_1850-2014.csv"))
dCo_data = LinearInterpolation(xs, df[!, "fgco2"][2:len+1])
dCa_data = LinearInterpolation(xs, df[!, "co2_inPgC"][2:len+1])
dCl_data = LinearInterpolation(xs, df[!, "netAtmosLandCO2Flux"][2:len+1])
#Iem = df[!, "dC_ocean"] + df[!, "dC_land"] + df[!, "dC_atmos"] # emission
#Iem_data = LinearInterpolation(xs, Iem[2:len+1])
#%%
Iem_data = LinearInterpolation(xs, df[!, "emission"][2:len+1])

#%% prepare parameeter container
pt = [α, β, σ, η, λ, γ, τ, Q10]

#%% define mechanistic model
function climate_carbon_cycle(du, u, p, t)
  Hp, Ho, H, To, T, Cl, Co, Ca = u # state variables
  α, β, σ, η, λ, γ, τ, Q10 = p # parameters + forcing

  CO2 = Ca * PgC_to_ppm # convert atmospheric carbon to atm. CO2 concentration + add forcing
  F = α*log(CO2/CO2_ref) # radiative forcing based on atm. CO2 concentration
  ΔT = T - T_ref # change in temperature

  ## energy
  du[1] = dHp = -(λ * ΔT) + F # planetary energy uptake (accum. imbalance)
  du[2] = dHo = γ*(ΔT - To) # deep-ocean energy uptake
  du[3] = dH = dHp - dHo # atmosphere/land/upper-ocean energy uptake
  du[4] = dTo = dHo / Co  # deep-ocean water temperature
  du[5] = dT = dH / C  # surface air temperature

  ## carbon
  du[6] = dCl = dCl_data[t] # NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/τ*Q10^(ΔT/10) # land carbon
  du[7] = dCo = dCo_data[t] # ocean carbon σ*((Ca - Ca_ref) - η*(Co - Co_ref))
  du[8] = dCa = - dCl - dCo + Iem_data[t] # atmos carbon

end

# %% solve the problem
# run solver
prob_true = ODEProblem(climate_carbon_cycle, u0, tspan, pt)
sol_true = solve(prob_true, Tsit5(), saveat=1)

#%% make plot
pl = plot(sol_true, layout=(4,2), label="",
          ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
          title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
plot!(pl, size=(750,1000))
savefig("plots/carbon+two-layer-ebm/Case00_with-1pctCO2-forcing.pdf")

#%% define hybrid model
function climate_carbon_cycle_hybrid(du, u, p, t)
  Hp, Ho, H, To, T, Cl, Co, Ca = u # state variables
  λ, β, Q10 = p # parameter to learn

  CO2 = Ca * PgC_to_ppm # convert atmospheric carbon to atm. CO2 concentration + add forcing
  F = α*log(CO2/CO2_ref) # radiative forcing based on atm. CO2 concentration
  ΔT = T - T_ref # change in temperature

  ## energy
  du[1] = dHp = -(λ * ΔT) + F # planetary energy uptake (accum. imbalance)
  du[2] = dHo = γ*(ΔT - To) # deep-ocean energy uptake
  du[3] = dH = dHp - dHo # atmosphere/land/upper-ocean energy uptake
  du[4] = dTo = dHo / Co  # deep-ocean water temperature
  du[5] = dT = dH / C  # surface air temperature

  ## carbon
  du[6] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/τ*Q10^(ΔT/10) # land carbon dCl_data[t]
  du[7] = dCo = dCo_data[t] # ocean carbon σ*((Ca - Ca_ref) - η*(Co - Co_ref))
  du[8] = dCa = - dCl - dCo + Iem_data[t] # atmos carbon

end

#%% prepare parameters
p = [1.0, 1.0, 1.0]

#%% solve the hybrid problem
prob_hybrid = ODEProblem(climate_carbon_cycle_hybrid, u0, tspan, p)
sol_hybrid = solve(prob_hybrid, Tsit5(), p=p, saveat=1)

#%% make plot before training
pl = plot(sol_true, layout=(4,2), label="true",
          ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
          title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
plot!(pl,sol_hybrid, layout=(4,2), label="prediction")      
plot!(pl, size=(750,1000))
savefig("plots/carbon+two-layer-ebm/Case01_lambda-beta-Q10_with-1pctCO2-forcing.pdf")

#%% set up training
opt = ADAM(0.01)
#opt = BFGS(initial_stepnorm = 0.1)
t = range(tspan[1],tspan[2],length=len)

function predict_true()
  Array(solve(prob_true, Tsit5(), saveat=1))
end

true_data = [predict_true()[1,:],
            Ca_ref .+ cumsum(df[!, "co2_inPgC"][2:len+1])]

function predict_hybrid(p)
  Array(solve(prob_hybrid, Tsit5(), p=p, saveat=t))
end

function loss_mse(p)
    pdata = predict_hybrid(p)
    tdata = true_data #predict_true()
    loss = Flux.mse([pdata[1,:];pdata[8,:]],[tdata[1,:][1];tdata[2,:][1]])
    #loss = Flux.mse(pdata[8,:],tdata)

    loss
end

losses = []
callback(θ,l) = begin
  push!(losses, l)
  if length(losses)%10==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  false
end
loss_mse(p)

# cb = function (;doplot=false) #callback function to observe training
#   pdata = predict_hybrid()
#   display(loss_mse())
#   # plot current prediction against data
#   pl = plot(tdata, layout=(4,2), label="data",
#           ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
#           title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
#   plot!(pl,pdata, layout=(4,2), label="prediction")      
#   plot!(pl, size=(750,1000))
#   display(plot(pl))
#   return false
# end

#%%
#p = [res.minimizer[1:length(p_init)]; 0.5]
res = DiffEqFlux.sciml_train(loss_mse, p, opt, maxiters= 1500, cb = callback)

#%% make plot after training
sol_hybrid_new = solve(prob_hybrid, Tsit5(), p=res.minimizer, saveat=t)

pl = plot(sol_true, layout=(4,2), label="true",
          ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
          title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
plot!(pl,sol_hybrid_new, layout=(4,2), label="prediction")      
plot!(pl, size=(750,1000))
savefig("plots/carbon+two-layer-ebm/Case01_lambda-beta-Q10_with-1pctCO2-forcing-after-training.pdf")

#%% test extrapolation
p_new = res.minimizer

# different init conditions
u0_02 = [0, 0, 0, 0, T_ref+5, Cl_ref, Co_ref, Ca_ref] # Hp, Ho, H, T, To, Cl, Co, Ca

# true solution
# run solver
prob_true_02 = ODEProblem(climate_carbon_cycle, u0_02, tspan, pt)
sol_true_02 = solve(prob_true_02, Tsit5(), p=pt, saveat=1)

# hybrid problem with trained network
prob_hybrid_02 = ODEProblem(climate_carbon_cycle_hybrid, u0_02, tspan, p_new)
sol_hybrid_02 = solve(prob_hybrid_02, Tsit5(), p=p_new, saveat=1)

#%% make plot before training
pl = plot(sol_true_02, layout=(4,2), label="true",
          ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
          title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
plot!(pl,sol_hybrid_02, layout=(4,2), label="prediction")      
plot!(pl, size=(750,1000))

#### Use NN in the hybrid model
#%% define neural model
# set up NN
#NN = FastChain(FastDense(3, 32, tanh), FastDense(32, 32, tanh), FastDense(32,1))
NN = FastChain(FastDense(2, 32, tanh), FastDense(32, 32, tanh), FastDense(32,1))
p_random = Float32.(initial_params(NN))
p = p_random
#NN([1,2,3],p_random)
NN([1,2],p_random)

#%% define hybrid model with neural network
function climate_carbon_cycle_hybrid_NN(du, u, p, t)
  Hp, Ho, H, To, T, Cl, Co, Ca = u # state variables
  #λ, β, Q10 = p # parameter to learn

  CO2 = Ca * PgC_to_ppm # convert atmospheric carbon to atm. CO2 concentration + add forcing
  F = α*log(CO2/CO2_ref) # radiative forcing based on atm. CO2 concentration
  ΔT = T - T_ref # change in temperature

  ## energy
  du[1] = dHp = -(λ * ΔT) + F # planetary energy uptake (accum. imbalance)
  du[2] = dHo = γ*(ΔT - To) # deep-ocean energy uptake
  du[3] = dH = dHp - dHo # atmosphere/land/upper-ocean energy uptake
  du[4] = dTo = dHo / Co  # deep-ocean water temperature
  du[5] = dT = dH / C  # surface air temperature

  ## carbon
  #du[6] = dCl = NN([CO2/CO2_ref,T/T_ref,Cl/Cl_ref],p)[1]
  du[6] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - NN([T/T_ref,Cl/Cl_ref],p)[1] # land carbon dCl_data[t]
  #du[6] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/τ*Q10^(ΔT/10) # land carbon dCl_data[t]
  du[7] = dCo = dCo_data[t] # ocean carbon σ*((Ca - Ca_ref) - η*(Co - Co_ref))
  du[8] = dCa = - dCl - dCo + Iem_data[t] # atmos carbon

end

#%% solve the hybrid problem
prob_hybrid = ODEProblem(climate_carbon_cycle_hybrid_NN, u0, tspan, p)
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))
sol_hybrid = solve(prob_hybrid, TRBDF2(autodiff=false), p=p, saveat=1, sensealg = sensealg)

#%% make plot
pl = plot(sol_true, layout=(4,2), label="true",
          ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
          title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
plot!(pl,sol_hybrid, layout=(4,2), label="prediction")      
plot!(pl, size=(750,1000))
savefig("plots/carbon+two-layer-ebm/Case02_Reco-as-NN_with-1pctCO2-forcing.pdf")

#savefig("Case02_learn-R_init.pdf")
#%% set up training
#opt = BFGS(initial_stepnorm = 0.1)
opt = ADAM(0.01)
t = range(tspan[1],tspan[2],length=len)

function predict_true()
  Array(solve(prob_true, Tsit5(), saveat=1))
end

true_data = [predict_true()[1,:],
            Ca_ref .+ cumsum(df[!, "co2_inPgC"][2:len+1])]

function predict_hybrid(p)
  Array(solve(prob_hybrid, Tsit5(), p=p, saveat=t, sensealg = sensealg))
end

function loss_mse(p)
    pdata = predict_hybrid(p)
    tdata = true_data #predict_true()
    loss = Flux.mse([pdata[1,:];pdata[8,:]],[tdata[1,:][1];tdata[2,:][1]])
    #loss = Flux.mse(pdata[8,:],tdata)

    loss
end

function predict(θ)
  temp_hybridprob = remake(prob_hybrid;p=θ,saveat=1)
  Array(solve(temp_hybridprob, RK4();reltol=1e-8,abstol=1e-8,sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

losses = []
callback(θ,l) = begin
  push!(losses, l)
  if length(losses)%10==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
  end
  false
end
loss_mse(p)

#%% train
#data = Iterators.repeated((), 1000)
#Flux.train!(loss_mass_balance, ps, data, opt, cb = cb)
#Flux.train!(loss_mse, ps, data, opt)

#%%
oopz = DiffEqFlux.sciml_train(loss_mse, p_random, opt,
                              maxiters = 100, cb = callback,
                              allow_f_increases = true)

#%% make plot after training
sol_hybrid_new = solve(prob_hybrid, Tsit5(), p=oopz.u, saveat=t)

# make plot after training
pl = plot(sol_true, layout=(4,2), label="true",
          ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
          title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
plot!(pl,sol_hybrid_new, layout=(4,2), label="prediction")      
plot!(pl, size=(750,1000))
savefig("plots/carbon+two-layer-ebm/Case02_Reco-as-NN_with-1pctCO2-forcing-after-training.pdf")


#%% test extrapolation
p_new = oopz.u

# different init conditions
u0_03 = [0, 0, 0, 0, T_ref, Cl_ref, Co_ref, Ca_ref] # Hp, Ho, H, To, T, Cl, Co, Ca

Iem_data = zero(Iem_data) .+ 10
Iem_data = LinearInterpolation(xs, Iem_data)

# run solver
prob_true = ODEProblem(climate_carbon_cycle, u0_03, tspan, pt)
sol_true = solve(prob_true, Tsit5(), p=pt, saveat=1)

# solve the hybrid problem
prob_hybrid = ODEProblem(climate_carbon_cycle_hybrid_NN, u0_03, tspan, p_new)
sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))
sol_hybrid = solve(prob_hybrid, TRBDF2(autodiff=false), p=p_new, saveat=1, sensealg = sensealg)

# make plot
pl = plot(sol_true, layout=(4,2), label="true",
          ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
          title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
plot!(pl,sol_hybrid, layout=(4,2), label="prediction")      
plot!(pl, size=(750,1000))
