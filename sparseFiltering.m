function [optW] = sparseFiltering(N, X)
    
    % N = # features, X = input data (examples in column)
    optW = randn(N, size(X, 1));
    optW = minFunc(@SparseFilteringObj, optW(:), ...
                   struct('MaxIter', 200, 'Corr', 20), X, N);
    optW = reshape(optW, [N, size(X, 1)]);
    
end

function [Obj, DeltaW] = SparseFilteringObj (W, X, N)
    % Reshape W into matrix form
    W = reshape(W, [N, size(X,1)]);
    
    % Feed Forward
    F = W*X;
    Fs = sqrt(F.^2 + 1e-8);
    [NFs, L2Fs] = l2row(Fs);
    [Fhat, L2Fn] = l2row(NFs');
    
    % Compute Objective Function
    Obj = sum(sum(Fhat, 2), 1);
    
    % Backprop through each feedforward step
    DeltaW = l2rowg(NFs', Fhat, L2Fn, ones(size(Fhat)));
    DeltaW = l2rowg(Fs, NFs, L2Fs, DeltaW');
    DeltaW = (DeltaW .* (F./Fs)) * X';
    DeltaW = DeltaW(:);
end

%
% The following functions are now in common/
%
% function [Y,N] = l2row(X) % L2 Normalize X by rows
%     % We also use this to normalize by column with l2row(X')
%     N = sqrt(sum(X.^2,2) + 1e-8);
%     Y = bsxfun(@rdivide,X,N);
% end
% 
% function [G] = l2grad(X,Y,N,D) % Backpropagate through Normalization
%     G = bsxfun(@rdivide, D, N) - bsxfun(@times, Y, sum(D.*X, 2)./(N.^2));
% end
% 
