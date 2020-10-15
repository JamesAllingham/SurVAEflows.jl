abstract type Transform end

abstract type StochasticTransform <: Transform end

abstract type Surjection <: Transform end

abstract type Bijection <: Transform end

@enum Direction inf gen

struct Augmentation <: Surjection
    encoder
    split_size
end

struct CouplingBijection <: Bijection end

struct Reverse <: Bijection end

struct Sequence <: Transform end
