abstract type Rounding <: Surjection end

struct GenRounding <: Rounding
    dist::Distribution
end

function forward(g::GenRounding, z) # TODO: Add types to function args?
    x = floor(z) # TODO: explicity cast to int?
end  # function forward

function inverse(g::GenRounding, x)
    z = x + rand(g.dist, size(x)) 
end  # function inverse

function V(g::GenRounding, x, z)
    -logpdf(g.dist, z - x)
end   # function V

struct InfRounding <: Rounding
    dist::Distribution
end

function forward(i::InfRounding, z)
    x = z + rand(i.dist, size(z))
end  # function forward

function inverse(i::InfRounding, x)
    z = floor(x)
end  # function inverse

function V(i::InfRounding, x, z)
    logpdf(i.dist, x - z)
end  # function V

# TODO: create Dequantization as an alias (of sorts???) for generative direction rounding?

# TODO: make sure that all Distributions' logpdf are compatible with backprop