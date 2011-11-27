function R = feedForwardSF(W, X)
    % Feed Forward
    F = W*X;
    Fs = sqrt(F.^2 + 1e-8);
    [NFs] = l2row(Fs);
    [Fhat] = l2row(NFs');
    R = Fhat';
end
