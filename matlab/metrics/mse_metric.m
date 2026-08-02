function m = mse_metric(X, Y)
%MSE_METRIC  Mean squared error.  MSE = (1/MN) * sum (X - Y)^2.
%
%   Both inputs are cast to double first: (X-Y).^2 on uint8 would saturate.

X = double(X); Y = double(Y);
m = mean((X(:) - Y(:)).^2);
end
