using LinearAlgebra, Plots, Distributions, Random, StableRNGs, Statistics

mutable struct neuron
    Act::Float64 
    Ge::Float64
    Gl::Float64
    GbarE::Float64
    GbarL::Float64
    ErevE::Float64
    ErevL::Float64
    ErevK::Float64
    theta::Float64
    Vm::Float64
    ΔVm::Float64
    GKNA::Float64   #for KNA adapt
    Tau::Float64    #for KNA adapt 
    Rise::Float64   #for KNA adapt
    Max::Float64    #for KNA adapt
    

    function neuron(GbarE, GbarL, ErevE, ErevL, ErevK, Vm, Tau, Rise, Max) 
        Act = 0
        Ge = 0
        Gl = GbarL
        ΔVm = 0
        theta = 0.5
        GKNA = 0
        
        return new(Act, Ge, Gl, GbarE, GbarL, ErevE, ErevL,ErevK, theta, Vm, ΔVm, GKNA , Tau, Rise, Max)
    end
end


function compute_g_theta_e(n::neuron)
    num = n.Gl * (n.ErevL - n.theta) + n.GKNA * (n.ErevK - n.theta)
    den = n.theta - n.ErevE
    return num / den
end

# Noisy xx1 function taken translated from https://github.com/emer/leabra/blob/master/nxx1/nxx1.go
# 

mutable struct Params
    Thr::Float64
    Gain::Float64
    NVar::Float64
    VmActThr::Float64
    SigMult::Float64
    SigMultPow::Float64
    SigGain::Float64
    InterpRange::Float64
    GainCorRange::Float64
    GainCor::Float64
    SigGainNVar::Float64
    SigMultEff::Float64
    SigValAt0::Float64
    InterpVal::Float64
    
end

function xx1(x::Float64)
    return x / (x + 1)
end

function xx1_gain_cor(xp::Params, x::Float64)
    gain_cor_fact = (xp.GainCorRange - (x / xp.NVar)) / xp.GainCorRange
    if gain_cor_fact < 0
        return xx1(xp.Gain * x)
    end
    new_gain = xp.Gain * (1 - xp.GainCor * gain_cor_fact)
    return xx1(new_gain * x)
end

function noisy_xx1(xp::Params, x::Float64)
    if x < 0
        return xp.SigMultEff / (1 + exp(-(x * xp.SigGainNVar)))
    elseif x < xp.InterpRange
        interp = 1 - ((xp.InterpRange - x) / xp.InterpRange)
        return xp.SigValAt0 + interp * xp.InterpVal
    else
        return xx1_gain_cor(xp, x)
    end
end

function create_params(noise::Float64 = 0.01)
    p = Params(0.5, 100, noise, 0.01, 0.33, 0.8, 3, 0.01, 10, 0.1, 0, 0, 0, 0)
    update_params!(p)
    return p
end

function update_params!(p::Params)
    p.SigGainNVar = p.SigGain / p.NVar
    p.SigMultEff = p.SigMult * (p.Gain * p.NVar) ^ p.SigMultPow
    p.SigValAt0 = 0.5 * p.SigMultEff
    p.InterpVal = xx1_gain_cor(p, p.InterpRange) - p.SigValAt0
end