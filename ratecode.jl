using LinearAlgebra, Plots, Distributions, Random, StableRNGs, Statistics
include("utilities.jl")

global seed = 1234
Random.seed!(seed)

xx1_noise = 0.01 # noise levels outside the range of 0.005 and 0.015 produce strange results
pr = create_params(xx1_noise)

# Plot XX1 function
x_values = range(-0.02, stop=0.04, length=1000)
noisy_xx1_values = [noisy_xx1(pr, x) for x in x_values]
xx1plot = plot(x_values, noisy_xx1_values, xlabel="x", ylabel="Smoothed XX1", legend=false)

# Neuron parameters, most important to change is GbarE 
# With current parameters G theta E varies between roughly 0.12 to 0.25
GbarE = 0.2; GbarL = 0.3; ErevE = 1; ErevL = 0.3; ErevK = -0.2; Vm = 0.3
Tau = 200; Rise = 0.02; Max = 0.2 # for KNa Adapt
spiking = false # set spiking model (true) or ratecode model (false)
KNaAdapt = true 

n = neuron(GbarE, GbarL,ErevE, ErevL, ErevK, Vm, Tau, Rise, Max)

n_cycles = 300
GeNoise = 0.02 # Noise for Ge

GePlot = Vector{Float64}(undef, n_cycles)
VmPlot = Vector{Float64}(undef, n_cycles)
ΔVmPlot = Vector{Float64}(undef, n_cycles)
GKNAPlot = Vector{Float64}(undef, n_cycles)
ActPlot = Vector{Float64}(undef, n_cycles)
G_theta_ePlot = Vector{Float64}(undef, n_cycles)


for cycle in 1:n_cycles
    if cycle > 20 && cycle < 180
        n.Ge = n.GbarE + (randn() * GeNoise)
    else
        n.Ge = 0
    end
    
    n.ΔVm = n.Ge*(n.ErevE - n.Vm) + n.Gl*(n.ErevL - n.Vm) + n.GKNA*(n.ErevK - n.Vm) # update current
    n.Vm += n.ΔVm / 10 # change membrane potential based on current
    

    if !spiking
        n.Act += abs(n.ΔVm) * (noisy_xx1(pr, n.Ge - compute_g_theta_e(n)) - n.Act) # Nxx1 activation function 
        #n.Act += abs(n.ΔVm) * (sigmoid(n.Ge - compute_g_theta_e(n), 500) - n.Act) # Sigmoid activation function
    end

    if KNaAdapt && !spiking
        n.GKNA += n.Act * n.Rise * (n.Max - n.GKNA) - (1/n.Tau * n.GKNA) # formula from O'Reily
    end

    if n.Vm > n.theta && spiking # spiking
        n.Vm = 0.3
        n.Act = 1
        if KNaAdapt
            n.GKNA += (n.Rise * (n.Max - n.GKNA))
        end
    elseif KNaAdapt
        n.GKNA -= ((1/n.Tau) * n.GKNA)
    end

    
    GePlot[cycle] = n.Ge
    VmPlot[cycle] = n.Vm
    ΔVmPlot[cycle] = n.ΔVm
    GKNAPlot[cycle] = n.GKNA
    ActPlot[cycle] = n.Act
    G_theta_ePlot[cycle] = compute_g_theta_e(n)

end

fig = plot(VmPlot, label="Vm")
fig = plot!(GePlot, label="Ge")
fig = plot!(ΔVmPlot, label="ΔVm")
fig = plot!(GKNAPlot, label="GKNA")
fig = plot!(ActPlot, label="Act")
fig = plot!(G_theta_ePlot, label="GthetaE")
fig = ylims!(-0.2, 1)


