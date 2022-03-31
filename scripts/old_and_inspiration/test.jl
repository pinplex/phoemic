# load packages
using DifferentialEquations
using Flux
using Plots
using DiffEqFlux

#%%
u0 = [0.8, 0.8]
tspan = (0.0f0,25.0f0)

ann = Chain(Dense(2,10,tanh), Dense(10,1))

p1,re = Flux.destructure(ann)
p2 = [-2.0,1.1]
p3 = [p1;p2]
ps = Flux.params(p3)

function dudt_(du,u,p,t)
    x, y = u
    du[1] = re(p[1:41])(u)[1]
    du[2] = p[end-1]*y + p[end]*x
end
prob = ODEProblem(dudt_,u0,tspan,p3)
concrete_solve(prob,Tsit5(),u0,p3,abstol=1e-8,reltol=1e-6)

#%%
function predict_adjoint()
  Array(concrete_solve(prob,Tsit5(),u0,p3,saveat=0.0:0.1:25.0))
end
loss_adjoint() = sum(abs2,x-1 for x in predict_adjoint())
loss_adjoint()

data = Iterators.repeated((), 300)
opt = ADAM(0.01)
iter = 0
cb = function ()
  global iter += 1
  if iter % 50 == 0
    display(loss_adjoint())
    display(plot(solve(remake(prob,p=p3,u0=u0),Tsit5(),saveat=0.1),ylim=(0,6)))
  end
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_adjoint, ps, data, opt, cb = cb)
