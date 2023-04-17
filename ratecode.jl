using LinearAlgebra, Plots, Distributions, Random, StableRNGs, Statistics
include("utilities.jl")

global seed = 1234

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

n = Vector{neuron}(undef, 2)
fig = Vector(undef, 2)

n[1] = neuron(GbarE, GbarL,ErevE, ErevL, ErevK, Vm, Tau, Rise, Max)
n[2] = neuron(GbarE, GbarL,ErevE, ErevL, ErevK, Vm, Tau, Rise, Max)

n_cycles = 300
GeNoise = 0.02 # Noise for Ge

GePlot = Matrix{Float64}(undef, n_cycles, 2)
VmPlot = Matrix{Float64}(undef, n_cycles, 2)
ΔVmPlot = Matrix{Float64}(undef, n_cycles, 2)
GKNAPlot = Matrix{Float64}(undef, n_cycles, 2)
ActPlot = Matrix{Float64}(undef, n_cycles, 2)
G_theta_ePlot = Matrix{Float64}(undef, n_cycles, 2)



for i in 1:2
    Random.seed!(seed)
    for cycle in 1:n_cycles
        if cycle > 20 && cycle < 180
            n[i].Ge = n[i].GbarE + (randn() * GeNoise)
        else
            n[i].Ge = 0
        end
        
        n[i].ΔVm = n[i].Ge*(n[i].ErevE - n[i].Vm) + n[i].Gl*(n[i].ErevL - n[i].Vm) + n[i].GKNA*(n[i].ErevK - n[i].Vm) # update current
        n[i].Vm += n[i].ΔVm / 10 # change membrane potential based on current
        
    
        if !spiking
            if i == 1
                n[i].Act += abs(n[i].ΔVm) * (noisy_xx1(pr, n[i].Ge - compute_g_theta_e(n[i])) - n[i].Act) # Nxx1 activation function 
            else
                n[i].Act += abs(n[i].ΔVm) * (sigmoid(n[i].Ge - compute_g_theta_e(n[i]), 500) * 0.8 - n[i].Act) # Sigmoid activation function
            end
        end
    
        if KNaAdapt && !spiking
            n[i].GKNA += n[i].Act * n[i].Rise * (n[i].Max - n[i].GKNA) - (1/n[i].Tau * n[i].GKNA) # formula from O'Reily
        end
    
        if n[i].Vm > n[i].theta && spiking # spiking
            n[i].Vm = 0.3
            n[i].Act = 1
            if KNaAdapt
                n[i].GKNA += (n[i].Rise * (n[i].Max - n[i].GKNA))
            end
        elseif KNaAdapt
            n[i].GKNA -= ((1/n[i].Tau) * n[i].GKNA)
        end
    
        
        GePlot[cycle, i] = n[i].Ge
        VmPlot[cycle, i] = n[i].Vm
        ΔVmPlot[cycle, i] = n[i].ΔVm
        GKNAPlot[cycle, i] = n[i].GKNA
        ActPlot[cycle, i] = n[i].Act
        G_theta_ePlot[cycle, i] = compute_g_theta_e(n[i])
    
    end

    fig[i] = plot(VmPlot[:,i], label="Vm")
    fig[i] = plot!(GePlot[:,i], label="Ge")
    fig[i] = plot!(ΔVmPlot[:,i], label="ΔVm")
    fig[i] = plot!(GKNAPlot[:,i], label="GKNA")
    fig[i] = plot!(ActPlot[:,i], label="Act")
    fig[i] = plot!(G_theta_ePlot[:,i], label="GthetaE")
    fig[i] = ylims!(-0.2, 1)

end

compare = plot(GePlot[:,1], label="Ge")
compare = plot!(ActPlot[:,1], label="Act nxx1")
compare = plot!(ActPlot[:,2], label="Act Sigmoid")


