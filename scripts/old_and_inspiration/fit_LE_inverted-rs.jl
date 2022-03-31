using NCDatasets
using Plots
using Flux
using Flux.Data: DataLoader
using Optim
using FeatureTransforms
using Statistics

#%% define constants
γ = 0.000665 * 101.325 # psychrometric constant; kPa C-1 (assuming P=101.325 kPa at sea level)
ρ_a = 1.225 # mean air density at constant pressure; kg m-3 for 15° C at sea level
c_p = 1.013e+3 # MJ KG-1 C-1; specific heat capacity or air at constant pressure

#%% load the data
#ds = NCDataset("../data/DE-Tha_preprocessed.nc")
ds = NCDataset("M:/people/relghawi/Lightning_Hybrid/analysis_2/DE-Tha.nc") # in the BGC network

LE = ds["LE_CORR_o"][:]#[1:100]
VPD = ds["VPD"][:]#[1:100]
ra = ds["ra"][:]#[1:100]
R_n = ds["NETRAD_o"][:]#[1:100]
G = ds["G_o"][:]#[1:100]
Δ = ds["delta"][:]#[1:100]
WD = ds["WD"][:]#[1:100]

#%% make overview plots
plot(
    scatter(R_n[1:100], LE[1:100], title="LE vs. R_n"),
    scatter(VPD[1:100], LE[1:100], title="LE vs. VPD"),
    scatter(G[1:100], LE[1:100], title="LE vs. G"),
    scatter(Δ[1:100], LE[1:100], title="LE vs. Δ"),
    scatter(ra[1:100], LE[1:100], title="LE vs. ra"),
    scatter(WD[1:100], LE[1:100], title="LE vs. WD"),
    )   

#%% calculate inverted rs
rs_inv = 1 ./ ((LE .* (1 ./ ra) .* γ ) ./ ( Δ .* (R_n .- G) .+ ρ_a .* c_p .* (1 ./ ra) .* VPD .- LE .* ( Δ .+ γ )))

#%% calculate LE inv
LE_inv = (Δ .* (R_n .- G) .+ ρ_a .* c_p .* VPD ./ ra) ./ (( 1 .+ ( rs_inv ./ ra )) .* γ .+ Δ)

#%% make histogram plot
#histogram(rs_inv[1:end])
histogram(LE[1:end])

#%% define hyperparameters
η = 0.001 # learning rate
opt = ADAM(η) # optimizer
#opt = BFGS(initial_stepnorm = 0.1)
n_epochs = 100 # numbers of epochs
n_batch = 48*14 # 2 weeks

#%% prepare data
Xdata = reduce(vcat, (VPD, R_n, WD)) #WD, G
#Xdata = reduce(hcat, (VPD, R_n, WD)) #WD, G
Ydata = LE # rs_inv

#Xdata = reshape(Float32.(Xdata), 3, 100)
Xdata = Float32.(Xdata)
Ydata = Float32.(Ydata)

#%% normalise the data
#Xdata = Flux.normalise(Xdata)
#xt = MeanStdScaling(Xdata)
#yt = MeanStdScaling(Ydata)

#Xdata = FeatureTransforms.apply(Xdata, xt)
#Ydata = FeatureTransforms.apply(Ydata, yt)

#plot(Xdata') # some somethi with normalisation

Xdata = Flux.normalise(Xdata)
plot(Xdata')

Ydata = Flux.normalise(Ydata)
plot(Ydata')

#%% split test and train data
Xtest = Xdata[:,1:5000]

# has to be same dimension array, not vector. Otherwise Flux.mse creates a matrix out of it
Ytest = Ydata[:,1:5000]

Xtrain = Xdata[:,5000:end]
Ytrain = Ydata[:,5000:end]

train_loader = DataLoader((Xtrain, Ytrain), batchsize=n_batch, shuffle=false);

#%% set up NN model
input_n = size(Xdata)[1]
model = Chain(
              Dense(input_n, 10, σ),
              Dense(10, 10, σ),
              Dense(10, 1)
              )
ps = Flux.params(model)

#%% define PM Monteith function
function calc_PM(x, model)
    VPD = x[1,:]
    ra = x[2,:]
    R_n = x[3,:]
    G = x[4,:]
    Δ = x[5,:]
    WD = x[6,:]
    (Δ .* (R_n .- G) .+ ρ_a .* c_p .* VPD ./ ra) ./ (( 1 .+ ( model(x) ./ ra )) .* γ .+ Δ)
end

#%% define loss functions
function loss_hybrid(x, y)
    #ŷ = model(x)
    #Flux.Losses.crossentropy(model(x), y)
    ŷ = calc_PM(x, model)
    Flux.Losses.mse(ŷ, y)
end

function loss(x, y)
    ŷ = model(x)
    Flux.mse(ŷ, y)
end

#%% define callback function
function evalcb()
    push!(alllosses,loss(Xtest, Ytest)) # collect loss
    push!(r_squared,Statistics.cor(model(Xtest)[:], Ytest[:])^2) # collect R^2
end

#%% loss and R^2 container
alllosses = []
r_squared = []

#%% run training loop
for epoch in 1:n_epochs
    println("Epoch $epoch")
    time = @elapsed Flux.train!(loss, ps, train_loader, opt, cb = evalcb)
    #println("Echo $epoch: $time secs")
    #@show epoch, loss(Xtest,Ytest)
end

#%% make plot
#scatter(calc_PM(Xtest, model)[:], Ytest[:])
scatter(model(Xtest)[:], Ytest[:])

#%% losses
plot(alllosses)
plot(r_squared)