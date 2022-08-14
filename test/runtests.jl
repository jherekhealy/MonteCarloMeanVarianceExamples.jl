using MonteCarloMeanVarianceExamples, Test
import Distributions,StatsBase
function testNormalLing(z, mRef, vRef, algo, rtolM, rtolV)
      s = SimulationStatistics{Float64}(algo)
        for zi in z
            recordValue(s,zi)
        end
        l = lastMeasure(s)
        m1 = mean(l)
        v1 = variance(l)
        @test isapprox(mRef, m1, rtol = rtolM)
        @test isapprox(vRef, v1, rtol = rtolV)
        return (m1/mRef-1.0, v1/vRef-1.0)
end

function testNormalLingAverage(;sorted=false)
      rng = MRG63k3a()
    dist = Distributions.Normal(100000.0,1.0)
    nit = 100
    mError = zeros(nit)
    vError = zeros(nit)
    mRef = zeros(nit)
    vRef = zeros(nit)
    algos = [Naive{Float64}(), ChanLewis{Float64}(), Ling{Float64}(), Kahan{Float64}(Naive{Float64}()), Kahan{Float64}(ChanLewis{Float64}()), Kahan{Float64}(Ling{Float64}()), Klein{Float64}(Naive{Float64}()),Klein{Float64}(Ling{Float64}()), Kahan{Float64}(Naive{Float64}(useShift=true)),Klein{Float64}(Naive{Float64}(useShift=true))]
    #using shift = first obs significantly improves naive algo for variance, although somewhat arbitrary first number. Ok with Kahan. => Alg 4? update paper?
    mList = Dict()
    vList = Dict()
    for algo in algos
        mList[algo] = Float64[]
        vList[algo] = Float64[]
    end
    for it=1:nit
    n = 100000
    z = [Distributions.quantile(dist, rand(rng)) for i = 1:n] #much much slower than randn(rng, n)
    if sorted
        sort!(z)
    end
    refAlgo = Naive{BigFloat}()
    s = SimulationStatistics{Float64}(refAlgo)
    for zi in z
        recordValue(s,zi)
    end
    l = lastMeasure(s)
    mRef[it] = mean(l)
    vRef[it] = variance(l)
    if it <= 3
        println(refAlgo, " ", it, " ", mRef[it], " ", vRef[it])
    end
    for algo in algos
      (mError, vError) = testNormalLing(z, mRef[it], vRef[it], algo, 1e-6, 1e-2)
      if it <= 3
          println(it, " ",algo," ",mError, " ",vError)
      end
      mList[algo] = push!(mList[algo],abs(mError))
      vList[algo] = push!(vList[algo],abs(vError))
    end
    end
     println("Reference ", StatsBase.mean(mRef)," ",StatsBase.mean(vRef))
    for algo in algos
        println(algo, " ", StatsBase.mean(mList[algo])," ",StatsBase.mean(vList[algo]))
    end
end

function priceBinaryAsset(spot, strike, vol, tte)
    dist = Distributions.Normal(0,1.0)
    return spot * Distributions.cdf(dist,  (log(spot / strike) + vol^2 * tte / 2) / (vol*sqrt(tte)))
end

struct BinaryAssetPayoff
    strike::Float64
    q::Float64
    rebate::Float64
end
evaluate(payoff::BinaryAssetPayoff, fS) = fS > payoff.strike ? payoff.q*fS : payoff.rebate
struct BinaryCashPayoff
    strike::Float64
    q::Float64
    rebate::Float64
end
evaluate(payoff::BinaryCashPayoff, fS) = fS > payoff.strike ? payoff.q : payoff.rebate

function testBinaryAsset(;sorted=false, n::Int=1000,nit::Int=10,payoff=BinaryAssetPayoff(1.5,1e6,0.0))
    spot = 1.0; strike = payoff.strike; vol = 0.5; tte=1.0; q = payoff.q
    println("analytic price ", priceBinaryAsset(spot,strike,vol,tte)*q)
    rng = MRG63k3a()
    dist = Distributions.Normal(0,1.0)
    algos = [Naive{BigFloat}(),Naive{Float64}(),Ling{Float64}(), Kahan{Float64}(Naive{Float64}(useShift=false)), Kahan{Float64}(ChanLewis{Float64}()), Kahan{Float64}(Ling{Float64}()), Klein{Float64}(Ling{Float64}()), Kahan{Float64}(Naive{Float64}(useShift=true)),Klein{Float64}(Naive{Float64}(useShift=true))]
    #using shift = first obs significantly improves naive algo for variance, although somewhat arbitrary first number. Ok with Kahan. => Alg 4? update paper?
    mList = Dict()
    vList = Dict()
    gList = Dict()
    for algo in algos
        mList[algo] = Float64[]
        vList[algo] = Float64[]
        gList[algo] = Float64[]
    end
    for it=1:nit

    z = [Distributions.quantile(dist, rand(rng)) for i = 1:n] #much much slower than randn(rng, n)
    if sorted
        sort!(z)
    end
    sqrtte = sqrt(tte)
        eps = 0.01
        rebate = 0.0
    futureSpots = @. spot * exp(-vol^2 *tte / 2 + vol*sqrtte*z)
    for algo in algos
        s = SimulationStatistics{Float64}(algo)
        for fS in futureSpots
            value = evaluate(payoff,fS)
            recordValue(s,value)
        end

    l = lastMeasure(s)
    m1 = mean(l)
    sv1 = sampleVariance(l)
    reset(s)
    futureSpotsShifted = futureSpots .* (1+eps)
    for fS in futureSpotsShifted
        value = evaluate(payoff,fS)
        recordValue(s,value)
    end
    l = lastMeasure(s)
    m1u = mean(l)
    reset(s)
    futureSpotsShifted = futureSpots .* (1-eps)
    for fS in futureSpotsShifted
        value = evaluate(payoff,fS)
        recordValue(s,value)
    end
    l = lastMeasure(s)
    m1d = mean(l)
    gamma1 = (m1u-2*m1+m1d)/(eps*eps*spot*spot);
    push!(mList[algo], m1)
    push!(vList[algo], sv1)
    push!(gList[algo], gamma1)
    end

    local mRef = 0.0;local svRef=0.0;local gammaRef = 0.0
    for (i,algo) in enumerate(algos)
        if i == 1
            mRef=  mList[algo][end]; svRef = vList[algo][end]; gammaRef = gList[algo][end]
            println(it, " ",algo, " ", mList[algo][end], " ",vList[algo][end]," ",gList[algo][end])
        else
            println(algo, " ", mList[algo][end], " ",vList[algo][end]," ",gList[algo][end], " ", mList[algo][end] - mRef, " ",vList[algo][end]- svRef," ",gList[algo][end]- gammaRef)
        end
    end
    end
    algo = algos[1]
    mRef = mList[algo]; svRef = vList[algo]; gammaRef = gList[algo]
    for (i,algo) in enumerate(algos)
            println(algo, " ",StatsBase.mean(abs.(mList[algo] .- mRef)), " ",StatsBase.mean(abs.(vList[algo] .- svRef))," ",StatsBase.mean(abs.(gList[algo] .- gammaRef)), " max ",maximum(abs.(mList[algo] .- mRef)), " ",maximum(abs.(vList[algo] .- svRef))," ",maximum(abs.(gList[algo] .- gammaRef)))
    end
    end
@testset "NormalLingSingle" begin
    rng = MRG63k3a()
    dist = Distributions.Normal(100000.0,1.0)
    nit = 100
    algos = [Naive{Float64}(), ChanLewis{Float64}(), Ling{Float64}(), Kahan{Float64}(Naive{Float64}()), Kahan{Float64}(ChanLewis{Float64}()), Kahan{Float64}(Ling{Float64}()), Klein{Float64}(Naive{Float64}()),Klein{Float64}(Ling{Float64}()), Kahan{Float64}(Naive{Float64}(useShift=true)),Klein{Float64}(Naive{Float64}(useShift=true))]
    n = 100000
    z = [Distributions.quantile(dist, rand(rng)) for i = 1:n] #much much slower than randn(rng, n)
    zSorted = sort(z)
    refAlgo = Naive{BigFloat}()
    s = SimulationStatistics{Float64}(refAlgo)
    for zi in z
        recordValue(s,zi)
    end
    l = lastMeasure(s)
    mRef = mean(l)
    vRef = variance(l)
    println(refAlgo,  " ", mRef, " ", vRef)
    for algo in algos
      (mError, vError) = testNormalLing(z, mRef, vRef, algo, 1e-6, 1e-2)
      println(algo," ",mError, " ",vError)
    end
    reset(s)
    for zi in zSorted
        recordValue(s,zi)
    end
    l = lastMeasure(s)
    mRef = mean(l)
    vRef = variance(l)
    println(refAlgo, " ordered ", mRef, " ", vRef)
    for algo in algos
      (mError, vError) = testNormalLing(zSorted, mRef, vRef, algo, 1e-6, 1e-2)
      println(algo," ordered ",mError, " ",vError)
    end

end



@testset "BinaryAssetUnsorted" begin
   testBinaryAsset(sorted=false, n=1000*1000)
end

@testset "BinaryAssetSorted" begin
   testBinaryAsset(sorted=true, n=1000*1000)
end

@testset "BinaryCashUnsorted" begin
   testBinaryAsset(sorted=false, n=1000*10000,nit=1,payoff=BinaryCashPayoff(1.5,1e6,0.0))
end

@testset "NormalLingAverageUnsorted" begin
  testNormalLingAverage(sorted=false)
end

@testset "NormalLingAverageSorted" begin
  testNormalLingAverage(sorted=true)
end
