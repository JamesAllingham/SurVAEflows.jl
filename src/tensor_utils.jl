# Utilities for manipulating tensors

function batchwise_sum(x)
    sum(reshape(x, (:, size(x)[end])), dims=1)
end  # function batchwise_sum

function batchwise_mean(x)
    mean(reshape(x, (:, size(x)[end])), dims=1)
end  # function batchwise_mean