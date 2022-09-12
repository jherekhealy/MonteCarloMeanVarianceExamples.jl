export SimulationStatistics,
    ChanLewis,
    Kahan,
    Ling,
    Klein,
    Knuth,
    Naive,
    recordValue,
    lastMeasure,
    variance,
    sampleVariance,
    mean,
    standardError


abstract type StatisticsAlgorithm end

mutable struct SimulationMeasure{T}
    index::Int
    mean::T
    variance::T
end

mutable struct SimulationStatistics{T}
    algo::StatisticsAlgorithm
    table::AbstractVector{SimulationMeasure{T}}
    nextSampleSize::Int
    SimulationStatistics{T}(salgo::StatisticsAlgorithm) where {T} =
        (reset(salgo); new(salgo, T[], 128 - 1))

end

Base.broadcastable(ss::SimulationStatistics) = Ref(ss)


mutable struct Ling{T} <: StatisticsAlgorithm
    counter::Int
    A::T
    Q::T
    Ling{T}() where {T} = new(0, zero(T), zero(T))
end

mutable struct ChanLewis{T} <: StatisticsAlgorithm
    counter::Int
    S::T
    A::T
    Q::T
    ChanLewis{T}() where {T} = new(0, zero(T), zero(T), zero(T))
end


mutable struct Naive{U} <: StatisticsAlgorithm
    counter::Int
    A::U
    Q::U
    K::U
    useShift::Bool
    Naive{U}(; useShift::Bool = false) where {U} =
        new(0, zero(U), zero(U), zero(U), useShift)
end

mutable struct Kahan{T,U} <: StatisticsAlgorithm
    parent::T
    Astar::U
    Qstar::U
    Kahan{U}(parent::T) where {T,U} = new{T,U}(parent, zero(U), zero(U))
end


mutable struct Knuth{T,U} <: StatisticsAlgorithm
    parent::T
    Astar::U
    Qstar::U
    Knuth{U}(parent::T) where {T,U} = new{T,U}(parent, zero(U), zero(U))
end

mutable struct Klein{T,U} <: StatisticsAlgorithm
    parent::T
    S::U
    CS::U
    CCS::U
    S2::U
    CS2::U
    CCS2::U
    Klein{U}(parent::T) where {T,U} =
        new{T,U}(parent, zero(U), zero(U), zero(U), zero(U), zero(U), zero(U))
end



function variance(sm::SimulationMeasure{T}) where {T}
    return sm.variance
end

function sampleVariance(sm::SimulationMeasure{T}) where {T}
    return sm.variance * sm.index / (sm.index - 1)
end

function standardError(sm::SimulationMeasure{T}) where {T}
    return sqrt(sampleVariance(sm) / sm.index)
end
function mean(sm::SimulationMeasure{T}) where {T}
    return sm.mean
end

function setNextSampleSize(ss::SimulationStatistics{T}) where {T}
    ss.nextSampleSize = size
end

function lastMeasure(ss::SimulationStatistics{T}) where {T}
    return lastMeasure(T, ss.algo)
end

function updateStatistics(ss::Ling{U}, x) where {U}
    ss.counter += 1
    ss.Q += ((ss.counter - 1) * (x - ss.A)^2) / (ss.counter)
    ss.A += (x - ss.A) / (ss.counter)
end


function Base.sum(ss::Ling{U}) where {U} 
    return ss.A * ss.counter
end

function updateStatistics(s::Kahan{Ling{T},U}, x) where {U,T}
    ss = s.parent
    ss.counter += 1
    z = ((ss.counter - 1) * (x - ss.A)^2) / (ss.counter)
    y = z - s.Qstar
    t = ss.Q + y
    s.Qstar = (t - ss.Q) - y
    ss.Q = t
    z = (x - ss.A) / ss.counter
    y = z - s.Astar
    t = ss.A + y
    s.Astar = (t - ss.A) - y
    ss.A = t
end


function updateStatistics(s::Klein{Ling{T},U}, x) where {U,T}
    ss = s.parent
    ss.counter += 1
    z = ((ss.counter - 1) * (x - ss.A) * (x - ss.A)) / (ss.counter)
    t = s.S2 + z
    c = abs(s.S2) >= abs(z) ? (s.S2 - t) + z : (z - t) + s.S2
    s.S2 = t
    t = s.CS2 + c
    cc = abs(s.CS2) >= abs(c) ? (s.CS2 - t) + c : (c - t) + s.CS2
    s.CS2 = t
    s.CCS2 += cc
    ss.Q = (s.S2 + s.CS2 + s.CCS2)

    z = (x - ss.A) / ss.counter
    t = s.S + z
    c = abs(s.S) >= abs(z) ? (s.S - t) + z : (z - t) + s.S
    s.S = t
    t = s.CS + c
    cc = abs(s.CS) >= abs(c) ? (s.CS - t) + c : (c - t) + s.CS
    s.CS = t
    s.CCS += cc
    ss.A = (s.S + s.CS + s.CCS)
end


function setShiftFromFirstObservation(ss::Naive{T}, x) where {T}
    if ss.useShift && ss.counter == 1
        ss.K = x
    end
end



function updateStatistics(s::Kahan{Naive{T},U}, x) where {U,T}
    ss = s.parent
    ss.counter += 1
    setShiftFromFirstObservation(ss, x)
    z = (convert(U, x) - ss.K)^2
    y = z - s.Qstar
    t = ss.Q + y
    s.Qstar = (t - ss.Q) - y
    ss.Q = t
    z = convert(U, x)
    y = z - s.Astar
    t = ss.A + y
    s.Astar = (t - ss.A) - y
    ss.A = t
end
function updateStatistics(s::Knuth{Naive{T},U}, b0) where {U,T}
    ss = s.parent
    ss.counter += 1
    setShiftFromFirstObservation(ss, b0)
    b = (convert(U, b0) - ss.K)^2
    ss.Q, ei = twoSum(ss.Q, b)
    s.Qstar += ei

    b = convert(U, b0)
    ss.A, ei = twoSum(ss.A, b)
    s.Astar += ei
end

function twoSum(a::T, b::T) where {T}
    x = a + b
    z = x - a
    e = (a - (x-z)) + (b-z)
    return (x,e)
end

function updateStatistics(s::Klein{Naive{T},U}, x) where {U,T}
    ss = s.parent
    ss.counter += 1
    setShiftFromFirstObservation(ss, x)
    z = (convert(U, x) - ss.K)^2 #the raw square implies a loss of accuracy
    t = s.S2 + z
    c = abs(s.S2) >= abs(z) ? (s.S2 - t) + z : (z - t) + s.S2
    s.S2 = t
    t = s.CS2 + c
    cc = abs(s.CS2) >= abs(c) ? (s.CS2 - t) + c : (c - t) + s.CS2
    s.CS2 = t
    s.CCS2 += cc
    ss.Q = (s.S2 + s.CS2 + s.CCS2)

    z = convert(U, x)
    t = s.S + z
    c = abs(s.S) >= abs(z) ? (s.S - t) + z : (z - t) + s.S
    s.S = t
    t = s.CS + c
    cc = abs(s.CS) >= abs(c) ? (s.CS - t) + c : (c - t) + s.CS
    s.CS = t
    s.CCS += cc
    ss.A = (s.S + s.CS + s.CCS)    
end



function Base.sum(ss::Knuth{Naive{U}}) where {U} 
    return ss.parent.A+ss.Astar
end

function Base.sum(ss::Naive{U}) where {U} 
    return ss.A
end

function Base.sum(ss::ChanLewis{U}) where {U} 
    return ss.S
end

function lastMeasure(::Type{T}, ss::Ling{U})::SimulationMeasure{T} where {T,U}
    return SimulationMeasure(ss.counter, convert(T, ss.A), convert(T, ss.Q / ss.counter))
end

function updateStatistics(ss::Naive{U}, x::T) where {U,T}
    ss.counter += 1
    setShiftFromFirstObservation(ss, x)
    ss.Q += (convert(U, x) - ss.K)^2
    ss.A += convert(U, x)
end

function updateStatistics(ss::ChanLewis{U}, x::T) where {U,T}
    ss.counter += 1
    ss.Q += ((ss.counter - 1) * (x - ss.A)^2) / (ss.counter)
    ss.S += x
    ss.A = ss.S / ss.counter
end


function updateStatistics(s::Kahan{ChanLewis{T},U}, x) where {U,T}
    ss = s.parent
    ss.counter += 1
    z = ((ss.counter - 1) * (x - ss.A)^2) / (ss.counter)
    y = z - s.Qstar
    t = ss.Q + y
    s.Qstar = (t - ss.Q) - y
    ss.Q = t
    y = x - s.Astar
    t = ss.S + y
    s.Astar = (t - ss.S) - y
    ss.S = t
    ss.A = ss.S / ss.counter
end


function lastMeasure(::Type{T}, ss::Union{Kahan,Klein})::SimulationMeasure{T} where {T}
    return lastMeasure(T, ss.parent)
end

function lastMeasure(::Type{T}, ss::ChanLewis)::SimulationMeasure{T} where {T}
    return SimulationMeasure(ss.counter, convert(T, ss.A), convert(T, ss.Q / ss.counter))
end

function Base.reset(ss::ChanLewis{T}) where {T}
    ss.counter = 0
    ss.A = zero(T)
    ss.Q = zero(T)
    ss.S = zero(T)
end

function Base.reset(s::Union{Kahan{U,T},Knuth{U,T}}) where {U,T}
    reset(s.parent)
    s.Qstar = zero(T)
    s.Astar = zero(T)
end

function Base.reset(ss::Union{Ling{T},Naive{T}}) where {T}
    ss.counter = 0
    ss.A = zero(T)
    ss.Q = zero(T)
end

function Base.reset(ss::Klein{U,T}) where {U,T}
    reset(ss.parent)
    ss.S = zero(T)
    ss.CS = zero(T)
    ss.CCS = zero(T)
    ss.S2 = zero(T)
    ss.CS2 = zero(T)
    ss.CCS2 = zero(T)
end

function lastMeasure(::Type{T}, ss::Naive{U})::SimulationMeasure{T} where {T,U}
    mean = ss.A / ss.counter
    return SimulationMeasure(ss.counter, T(mean), T(ss.Q / (ss.counter) - (mean - ss.K)^2))
end


function lastMeasure(::Type{T}, s::Knuth{Naive{U}})::SimulationMeasure{T} where {T,U}
    ss = s.parent
    mean = (ss.A+s.Astar)/ss.counter
    return SimulationMeasure(ss.counter, convert(T, mean), convert(T, (ss.Q+s.Qstar) / ss.counter - (mean - ss.K)^2))
end

function recordValue(ss::SimulationStatistics{T}, x) where {T}
    updateStatistics(ss.algo, x)
    if counter(ss) == ss.nextSampleSize
        push!(ss.table, lastMeasure(T, ss.algo))
        ss.nextSampleSize = 2 * (counter(ss) + 1) - 1
    end
end

function Base.sum(ss::Union{Klein{T,U},Kahan{T,U}}) where {T,U}
    return Base.sum(ss.parent)
end

function Base.sum(ss::SimulationStatistics{T}) where {T}
    return Base.sum(ss.algo)
end

function counter(ss::SimulationStatistics{T}) where {T}
    return counter(ss.algo)
end

function counter(ss::Union{Klein{T,U},Kahan{T,U},Knuth{T,U}}) where {T,U}
    return ss.parent.counter
end

function counter(ss::Union{Naive{T},ChanLewis{T},Ling{T}}) where {T}
    return ss.counter
end

function simulationMeasures(ss::SimulationStatistics{T}) where {T}
    return ss.table
end

function Base.reset(ss::SimulationStatistics{T}) where {T}
    reset(ss.algo)
    ss.table = T[]
    ss.nextSampleSize = 128 - 1
end
