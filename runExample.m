%% Startup (addpaths)
startup

%% Clear
clear ; close all ; clc ;

%% Load Data
%  You can obtain patches.mat from 
%  http://cs.stanford.edu/~jngiam/data/patches.mat

fprintf('Loading Data\n');

%  Loads a variable data (size 256x50000)
load patches.mat

%% Remove DC
data = bsxfun(@minus, data, mean(data));

%% Train Layer 1
L1_size = 256;  % Increase this for more features
L1 = sparseFiltering(L1_size, data);

% Show Layer 1 Bases
displayData(L1);
pause;

%% Feed-forward Layer 1
data1 = feedForwardSF(L1, data);
data1 = bsxfun(@minus, data1, mean(data1));

%% Train Layer 2
L2_size = 256;
L2 = sparseFiltering(L2_size, data1);

%% Visualize Layer 2
figure;

% Number of L2 units to visualize
num_viz = 10; 

% Visualize different units
offset = 1;

% Plot the units
for i = 1:num_viz
    j = offset+i;
    
    % Find the sign of the unit with the maximum absolute values
    [a, b] = max(abs(L2(j, :)));
    sgn    = sign(L2(j, b)); 
    
    % Sort and plot units in those direction
    [a, b] = sort(sgn*(L2(j, :)), 'descend');
    
    % Plot
    subplot(1, num_viz, i, 'align'); 
    displayData(L1(b(1:10), :), [] , 1);
end
