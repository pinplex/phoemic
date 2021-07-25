# load packages
using DifferentialEquations
using Flux
using Flux.Data: DataLoader
using Plots
using LaTeXStrings
using DiffEqFlux
using Interpolations
using ForwardDiff

#%% parameters
# constants
dt = 60. * 60. * 24. * 365.   # one year expressed in seconds

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

#%% make forcing timeline
c = 280
forcing = Array{Float64}(undef, len+1) #or Float64 or Any instead of Int64
for i in 1:len+1
    c = c * 1.01
    forcing[i]= c
end

forcing = forcing .- 280
forcing = diff(forcing)
xs = range(1, length=len)
g = LinearInterpolation(xs, forcing)

#%% prepare parameeter container
p = [α, β, σ, η, λ, γ, τ, Q10, g]

#%% define mechanistic model
function climate_carbon_cycle(du, u, p, t)
  Hp, Ho, H, To, T, Cl, Co, Ca = u # state variables
  α, β, σ, η, λ, γ, τ, Q10, g = p # parameters

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
  du[6] = dCl = NPP_ref*(1 + β*log(CO2/CO2_ref)) - Cl/τ*Q10^(ΔT/10) # land carbon
  du[7] = dCo = σ*((Ca - Ca_ref) - η*(Co - Co_ref)) # ocean carbon
  du[8] = dCa = - dCl - dCo + g[t]/PgC_to_ppm # atmos carbon

end

# %% solve the problem
# run solver
prob = ODEProblem(climate_carbon_cycle, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=1)

#%% make plot
pl = plot(sol, layout=(4,2), label="",
          ylabel=[L"W m^{-2}" L"W m^{-2}" L"W m^{-2}" "K" "K" "Pg C" "Pg C" "Pg C"],
          title=["E_planet" "E_ocean" "E_atmos" "T_ocean" "T_atmos" "C_land" "C_ocean" "C_atmos" ])
plot!(pl, size=(750,1000))
#savefig("../plots/carbon+two-layer-ebm/Case00_plus-2xCO2.pdf")
