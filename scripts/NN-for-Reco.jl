# load packages
using Flux
using Flux.Data: DataLoader
using CSV
using DataFrames
using Plots

#%% define global variables
home = pwd()


#%% load data
df = DataFrame(CSV.File(home*"/data/carbon-budget_MPI-ESM1-2-LR_1pctCO2_1850-2014.csv"))
tas_data = df[!, "tas"][2:len+1]
Cland_data = df[!, "cLand"][2:len+1]
reco_data = df[!, "reco"][2:len+1]

# calc T delta
tas_data = tas_data .- tas_data[1]
reco_data = reco_data .- reco_data[1]

#%% define hyperparameters
η = 0.001 # learning rate
opt = ADAM(η) # optimizer
#opt = BFGS(initial_stepnorm = 0.1)
n_epochs = 10000 # numbers of epochs
n_batch = 140

#%% prepare data
n = length(reco_data)
#x = reshape(hcat(tas_data,Cland_data), n, 2)
x = reduce(hcat, (tas_data,Cland_data))
Xdata = Float32.(x)
Ydata = Float32.(reco_data)

#Xdata = Flux.normalise(Xdata)

Xtest = Xdata'
Ytest = Ydata

Xtrain = Xdata'
Ytrain = Ydata

train_loader = DataLoader((Xtrain, Ytrain), batchsize=n_batch, shuffle=false);

#%% set up NN model
model = Chain(Dense(2, 20, tanh), Dense(20, 20, tanh), Dense(20, 1))

ps = Flux.params(model)

#%% define loss function
function loss(x, y)
    ŷ = model(x)
    return Flux.Losses.mse(ŷ, y)
end

#%% define callback function
cb = () -> push!(alllosses,loss(Xtest, Ytest))

#%% define training function
function my_custom_train!(loss, ps, data, opt)
    # declare training loss local so we can use it outside gradient calculation
    local training_loss                                                            
    ps = Params(ps)
    print(ps)                                                                
    for d in data
      #x = transpose(d[1])
      #y = transpose(d[2])                                                               
      gs = gradient(ps) do
        print(d)                                                            
        training_loss = loss(d...)
        return training_loss
      end  
      
      #print(ps)
      #print(gs)
      
      #Flux.update!(opt, ps, gs)                                                       
    end                                                                               
  end 

#%% losses container
alllosses = []

#%% run training loop
for epoch in 1:n_epochs
    @time Flux.train!(loss, ps, train_loader, opt, cb = cb)
end
#%% make plot
scatter(model(Xtest), Ytest)

#%% plot losses
plot(alllosses)


