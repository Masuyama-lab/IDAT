% Copyright (c) 2025-2026 Naoki Masuyama
% SPDX-License-Identifier: MIT

classdef IDAT < handle
    % IDAT  Inverse Distance-based ART with Topology.
    %
    %   net = IDAT();                 % a new model, with no hyperparameter to set
    %   net.fit(X);                   % learn the rows of X one at a time, in the given order
    %   labels = net.predict(Y);      % cluster label of every row of Y, as a 1 x size(Y, 1) vector
    %
    %   fit can be called again with new samples. The model keeps learning from where it stopped,
    %   so a data stream can be given chunk by chunk.
    %
    % How it works (one sample x at a time)
    %   Category choice   Every node k has a prototype w_k and a scale sigma_k. The similarity of
    %                     x to node k is s_k = 1 / (1 + ||x - w_k|| / sigma_k). The two most
    %                     similar nodes are s1 and s2. The first three samples become nodes.
    %   Vigilance test    V is the similarity threshold (the vigilance parameter).
    %                       s_1 <  V           : x becomes a new node.
    %                       s_1 >= V, s_2 <= V : w_s1 moves towards x.
    %                       s_1 >= V, s_2 >  V : w_s1 and w_s2 move towards x. When both nodes
    %                     have won more often than the average of the nodes that have won at
    %                     least twice, both become active (isWeight) and their co-activation
    %                     count E^cand(s1, s2) grows. An edge is created once that count exceeds
    %                     the average of the positive co-activation counts.
    %   Maintenance       Every Lambda samples: edges longer than Q3 + 1.5 IQR of the edge
    %                     lengths are removed, nodes that have never won since their creation
    %                     are removed except the newest Lambda of them, and an active node that
    %                     is isolated and has stayed idle for two maintenances becomes inactive.
    %   Adaptation        Every Lambda samples, once there are more than two nodes: Lambda is set
    %                     to the largest number L of recent samples whose similarity matrix S_L
    %                     is still diverse, det(S_L) >= EPS_DET, and V is set to a quantile of
    %                     the similarities within these samples. The quantile level follows the
    %                     ratio of the number of clusters to the number of nodes, smoothed over
    %                     the adaptations.
    %   Prediction        A sample gets the label of its most similar active node (of all nodes
    %                     if none is active). The labels are the connected components of the
    %                     edge graph.
    %
    % Properties (read only)
    %   numNodes               K, the number of nodes, active or not
    %   weight                 [K x d] prototypes
    %   countNode              [K x 1] number of wins of every node
    %   bufferStd              [K x 1] sigma_k
    %   isWeight               [K x 1] true for the active nodes, the nodes used by predict
    %   edge                   [K x K] sparse, edge(i, j) > 0 when nodes i and j are connected
    %   edgeCandidateCounter   [K x K] sparse, co-activation counts E^cand
    %   lambda                 current window length and maintenance interval
    %   similarityThreshold    current similarity threshold V
    %   labelCluster           [K x 1] cluster label of every node, set by predict
    %   sampleCounts           number of samples learned so far
    %
    % Network size (the three counts printed by the main scripts)
    %   all nodes      numNodes
    %   active nodes   nnz(isWeight)
    %   clusters       numel(unique(labelCluster(isWeight))) after predict, that is,
    %                  the clusters that contain at least one active node
    %
    % The constants below are the values used in the paper. They are not meant to be tuned.
    %
    % Requires the Statistics and Machine Learning Toolbox (pdist, pdist2, quantile, squareform).
    %
    % Reference
    %   N. Masuyama, Y. Toda, Y. Nojima, and H. Ishibuchi, "An adaptive resonance theory-based
    %   topological clustering algorithm with a self-adjusting vigilance parameter,"
    %   arXiv:2511.17983, 2025. https://arxiv.org/abs/2511.17983

    properties (Constant)
        LAMBDA_INIT      = 2         % initial Lambda
        EPS_DET          = 1.0e-6    % a window of L samples is diverse while det(S_L) >= EPS_DET
        IQR_FACTOR       = 1.5       % c of the bounds Q3 + c IQR (edge pruning) and Q1 - c IQR (demotion)
        DEMOTE_TOLERANCE = 2         % idle maintenances before an isolated active node is demoted
        BUFFER_FACTOR    = 2         % the buffer keeps the newest BUFFER_FACTOR * Lambda samples
        SIGMA_MIN        = 1.0e-6    % lower bound of sigma_k
    end

    properties (SetAccess = private)
        numNodes             = 0
        weight               = []
        countNode            = zeros(0, 1)
        bufferStd            = zeros(0, 1)
        isWeight             = false(0, 1)
        edge                 = sparse(0, 0)
        edgeCandidateCounter = sparse(0, 0)
        lambda
        similarityThreshold  = 0
        labelCluster         = zeros(0, 1)
        sampleCounts         = 0
    end

    properties (Access = private)
        recalcCounter    = 0            % samples learned since the last adaptation
        ratioNC          = 0            % smoothed ratio of the number of clusters to the number of nodes
        % ring buffer of the most recent samples
        buf              = []           % [bufCapacity x d]
        bufCapacity      = 0
        bufStart         = 1            % row of the oldest kept sample
        bufCount         = 0            % number of kept samples
        bufKeep          = 0            % samples to keep (0 before the first adaptation: all)
        % running mean, second moment and count of the learned samples (Welford)
        runMean          = []
        runM2            = []
        runCount         = 0
        % running sums behind the two thresholds of the vigilance test
        sumCountAbove1   = 0            % sum of the counts > 1
        numCountAbove1   = 0            % number of the counts > 1
        sumCand          = 0            % sum of the positive entries of edgeCandidateCounter
        numCand          = 0            % number of these entries
        % graph bookkeeping
        degree           = zeros(0, 1)  % [K x 1] number of edges of every node
        candRowSum       = zeros(0, 1)  % [K x 1] row sums of edgeCandidateCounter
        numComponents    = 0            % number of connected components (valid when ~componentsStale)
        componentsStale  = true
        % demotion
        prevCountNode    = zeros(0, 1)  % countNode at the last maintenance
        prevCandRowSum   = zeros(0, 1)  % candRowSum at the last maintenance
        idleMaintenances = zeros(0, 1)  % consecutive maintenances as a demotion candidate
    end

    methods
        function obj = IDAT()
            obj.lambda = IDAT.LAMBDA_INIT;
        end

        function obj = fit(obj, samples)
            % FIT  Learn the rows of samples one at a time, in the given order.
            samples = double(samples);
            for n = 1:size(samples, 1)
                x = samples(n, :);
                obj.appendToBuffer(x);
                obj.learnOneSample(x);
                obj.sampleCounts = obj.sampleCounts + 1;
                obj.recalcCounter = obj.recalcCounter + 1;
                if obj.recalcCounter >= obj.lambda && obj.numNodes > 2
                    obj.adaptLambdaAndThreshold();
                    obj.recalcCounter = 0;
                    obj.updateBufferKeepLen();
                end
            end
        end

        function labels = predict(obj, samples)
            % PREDICT  Cluster label of every row of samples, as a row vector.
            %   A sample gets the label of its most similar active node (of all nodes if none is
            %   active). The labels are the connected components of the edge graph, numbered
            %   from 1 in the order of their smallest node index.
            samples = double(samples);
            if obj.numNodes == 0
                labels = [];
                return;
            end
            obj.labelCluster = conncomp(graph(obj.edge ~= 0)).';
            if any(obj.isWeight)
                nodes = find(obj.isWeight);
            else
                nodes = (1:obj.numNodes).';
            end
            alpha = 1 ./ obj.bufferStd(nodes);
            D = sqrt(max(pdist2(obj.weight(nodes, :), samples, 'squaredeuclidean'), 0));
            S = 1 ./ (1 + alpha .* D);
            [~, winner] = max(S, [], 1);
            labels = reshape(obj.labelCluster(nodes(winner)), 1, []);
        end
    end

    methods (Access = private)
        function learnOneSample(obj, x)
            if obj.numNodes < 3
                obj.addNode(x);
                if obj.numNodes == 2
                    obj.bufferStd(1) = obj.bufferStd(2);   % the first node takes the scale of the first two samples
                end
            else
                % category choice: the two nearest nodes (min returns the first of equal distances)
                distances = sqrt(sum((obj.weight - x).^2, 2));
                [d1, s1] = min(distances);
                distances(s1) = NaN;                       % min ignores NaN
                [d2, s2] = min(distances);
                sim1 = 1 ./ (1 + (1 ./ obj.bufferStd(s1, 1)) .* d1);
                sim2 = 1 ./ (1 + (1 ./ obj.bufferStd(s2, 1)) .* d2);
                V = obj.similarityThreshold;
                if sim1 < V
                    obj.addNode(x);                                    % Case I
                else
                    obj.incrementCount(s1);                            % Case II
                    w = obj.weight(s1, :);
                    obj.weight(s1, :) = w + (1 / obj.countNode(s1)) * (x - w);
                    obj.bufferStd(s1, 1) = max(obj.updateRunningStd(x), IDAT.SIGMA_MIN);
                    if sim2 > V                                         % Case III
                        obj.incrementCount(s2);
                        w = obj.weight(s2, :);
                        obj.weight(s2, :) = w + (1 / (obj.countNode(s1) + obj.countNode(s2))) * (x - w);
                        obj.updateTopology(s1, s2);
                    end
                end
            end
            % maintenance every Lambda samples (sampleCounts is the number of samples learned before x)
            if mod(obj.sampleCounts, obj.lambda) == 0
                obj.maintain();
            end
        end

        function updateTopology(obj, s1, s2)
            % Activation of s1 and s2, their co-activation count and, possibly, an edge (Case III).
            countTh = obj.sumCountAbove1 / obj.numCountAbove1;         % mean of the counts > 1
            if obj.countNode(s1) > countTh && obj.countNode(s2) > countTh
                obj.isWeight(s1) = true;
                obj.isWeight(s2) = true;
                c = obj.edgeCandidateCounter(s1, s2);
                obj.edgeCandidateCounter(s1, s2) = c + 1;
                obj.edgeCandidateCounter(s2, s1) = c + 1;
                obj.candRowSum([s1 s2]) = obj.candRowSum([s1 s2]) + 1;
                obj.sumCand = obj.sumCand + 2;
                obj.numCand = obj.numCand + 2 * (c == 0);
                edgeTh = obj.sumCand / obj.numCand;                     % mean of the positive entries of E^cand
                if c + 1 > edgeTh
                    e = obj.edge(s1, s2);
                    if e == 0
                        obj.componentsStale = true;
                        obj.degree([s1 s2]) = obj.degree([s1 s2]) + 1;
                    end
                    obj.edge(s1, s2) = e + 1;
                    obj.edge(s2, s1) = e + 1;
                end
            end
        end

        function maintain(obj)
            % 1) edge pruning: Tukey's rule on the edge lengths
            [i, j] = find(obj.edge);                                    % every edge is listed as (i, j) and (j, i)
            if ~isempty(i)
                len = sqrt(sum((obj.weight(i, :) - obj.weight(j, :)).^2, 2));
                Q1 = quantile(len, 0.25);
                Q3 = quantile(len, 0.75);
                IQR = Q3 - Q1;
                tooLong = len > Q3 + IDAT.IQR_FACTOR * IQR;
                if any(tooLong)
                    K = size(obj.edge, 1);
                    lin = i(tooLong) + (j(tooLong) - 1) * K;
                    obj.edge(lin) = 0;
                    obj.edgeCandidateCounter(lin) = 0;
                    obj.refreshCandidateSums();
                    obj.componentsStale = true;
                    obj.degree = full(sum(obj.edge > 0, 2));
                    obj.candRowSum = full(sum(obj.edgeCandidateCounter, 2));
                end
            end
            % 2) removal of the nodes that have never won: the newest Lambda of them are kept
            numCount1 = obj.numNodes - obj.numCountAbove1;
            if obj.numCountAbove1 > 0 && numCount1 > obj.lambda
                r = find(obj.countNode == 1);
                obj.removeNodes(r(1:end - obj.lambda));
            end
            % 3) demotion of the isolated active nodes that stay idle
            obj.demoteIdleNodes();
        end

        function removeNodes(obj, r)
            % The nodes r have count 1, so they have no edge and no co-activation: every one of
            % them is an isolated component, and the degrees and row sums of the other nodes and
            % the sums behind the thresholds do not change.
            obj.edge(r, :) = [];
            obj.edge(:, r) = [];
            obj.edgeCandidateCounter(r, :) = [];
            obj.edgeCandidateCounter(:, r) = [];
            obj.weight(r, :) = [];
            obj.countNode(r) = [];
            obj.bufferStd(r) = [];
            obj.isWeight(r) = [];
            obj.degree(r) = [];
            obj.candRowSum(r) = [];
            obj.prevCountNode(r) = [];
            obj.prevCandRowSum(r) = [];
            obj.idleMaintenances(r) = [];
            obj.numNodes = size(obj.weight, 1);
            obj.numComponents = obj.numComponents - numel(r);
        end

        function demoteIdleNodes(obj)
            % The activity of a node is its number of wins and co-activations since the last
            % maintenance. An active node that is isolated and whose activity is at most
            % max(Q1 - 1.5 IQR, 0) of the positive activities for DEMOTE_TOLERANCE consecutive
            % maintenances is demoted (isWeight = false).
            activity = (obj.countNode - obj.prevCountNode) + (obj.candRowSum - obj.prevCandRowSum);
            obj.prevCountNode  = obj.countNode;
            obj.prevCandRowSum = obj.candRowSum;
            hasActivity = activity > 0;
            if ~any(hasActivity)
                obj.idleMaintenances(:) = 0;
                return;
            end
            isolatedActive = obj.isWeight & (obj.degree == 0);
            if any(isolatedActive)
                a = activity(hasActivity);
                Q1 = quantile(a, 0.25);
                Q3 = quantile(a, 0.75);
                IQR = Q3 - Q1;
                candidate = isolatedActive & (activity <= max(Q1 - IDAT.IQR_FACTOR * IQR, 0));
            else
                candidate = false(obj.numNodes, 1);
            end
            obj.idleMaintenances(candidate)  = obj.idleMaintenances(candidate) + 1;
            obj.idleMaintenances(~candidate) = 0;
            demote = obj.idleMaintenances >= IDAT.DEMOTE_TOLERANCE;
            obj.isWeight(demote) = false;
            obj.idleMaintenances(demote) = 0;
        end

        function adaptLambdaAndThreshold(obj)
            % New Lambda: the largest L such that det(S_L) >= EPS_DET, where S_L is the similarity
            % matrix of the L most recent samples. One Cholesky factorisation of the similarity
            % matrix of the whole window gives every det(S_L) as a product of squared pivots.
            % New V: the (1 - r) quantile of the largest similarity of every one of the Lambda most
            % recent samples to another one, where r is the smoothed cluster-to-node ratio.
            if obj.componentsStale
                obj.numComponents = obj.countComponents();
                obj.componentsStale = false;
            end
            lam   = obj.lambda;
            alpha = 1 / max(obj.bufferStd);
            L = min(IDAT.BUFFER_FACTOR * lam, obj.bufCount);
            W = obj.newestRows(L);                                     % the newest sample first
            S = 1 ./ (1 + alpha * squareform(pdist(W, 'euclidean')));
            [R, p] = chol(S, 'lower');                                  % p > 0: only the leading p - 1 rows were factorised
            if p == 0
                m = L;
            else
                m = p - 1;
            end
            dR = diag(R);
            detPrefix = cumprod(dR(1:m)) .^ 2;                          % det(S_L) for L = 1..m
            k = find(detPrefix < IDAT.EPS_DET, 1, 'first');
            if isempty(k)
                newLambda = m;
            else
                newLambda = k - 1;
            end
            if newLambda < 2
                return;                                                 % not even two samples are diverse: keep Lambda, V and r
            end
            if newLambda < lam
                beta = 1 - (1 - 1/lam)^(newLambda - 1);
            else
                beta = 1 / lam;
            end
            obj.ratioNC = beta * (obj.numComponents / obj.numNodes) + (1 - beta) * obj.ratioNC;
            Ssub = S(1:newLambda, 1:newLambda);
            Ssub(1:newLambda+1:end) = -inf;
            u = max(Ssub, [], 2);
            obj.lambda = newLambda;
            obj.similarityThreshold = quantile(u, 1 - obj.ratioNC);
        end

        function C = countComponents(obj)
            % Number of connected components of the edge graph: every node without an edge is one
            % component, and the nodes with an edge are counted by the Dulmage-Mendelsohn
            % decomposition of their adjacency pattern with a nonzero diagonal.
            withEdge = find(obj.degree > 0);
            if isempty(withEdge)
                C = obj.numNodes;
            else
                [~, ~, r] = dmperm(spones(obj.edge(withEdge, withEdge)) + speye(numel(withEdge)));
                C = (obj.numNodes - numel(withEdge)) + (numel(r) - 1);
            end
        end

        function addNode(obj, x)
            k = obj.numNodes + 1;
            obj.numNodes = k;
            obj.weight(k, :) = x;
            obj.countNode(k, 1) = 1;
            obj.bufferStd(k, 1) = max(obj.updateRunningStd(x), IDAT.SIGMA_MIN);
            obj.isWeight(k, 1) = false;
            obj.edge(k, k) = 0;                        % grows both sparse matrices by an empty row and column
            obj.edgeCandidateCounter(k, k) = 0;
            obj.degree(k, 1) = 0;
            obj.candRowSum(k, 1) = 0;
            obj.prevCountNode(k, 1) = 0;
            obj.prevCandRowSum(k, 1) = 0;
            obj.idleMaintenances(k, 1) = 0;
            obj.numComponents = obj.numComponents + 1; % a new node is an isolated component
        end

        function incrementCount(obj, k)
            c = obj.countNode(k);
            obj.countNode(k) = c + 1;
            if c == 1                                  % the node now has a count > 1
                obj.numCountAbove1 = obj.numCountAbove1 + 1;
                obj.sumCountAbove1 = obj.sumCountAbove1 + 2;
            else
                obj.sumCountAbove1 = obj.sumCountAbove1 + 1;
            end
        end

        function refreshCandidateSums(obj)
            v = nonzeros(obj.edgeCandidateCounter);
            obj.sumCand = sum(v);
            obj.numCand = numel(v);
        end

        function s = updateRunningStd(obj, x)
            % Welford's update of the running mean and variance of the learned samples; returns
            % the largest standard deviation over the features.
            if obj.runCount == 0
                obj.runMean  = x;
                obj.runM2    = zeros(size(x));
                obj.runCount = 1;
                s = 0;
            else
                obj.runCount = obj.runCount + 1;
                delta = x - obj.runMean;
                obj.runMean = obj.runMean + delta / obj.runCount;
                obj.runM2   = obj.runM2 + delta .* (x - obj.runMean);
                s = sqrt(max(obj.runM2 / (obj.runCount - 1)));
            end
        end

        function appendToBuffer(obj, x)
            if obj.bufCapacity == 0
                obj.bufCapacity = 500;
                obj.buf = zeros(obj.bufCapacity, numel(x));
            elseif obj.bufCount == obj.bufCapacity
                obj.growBuffer(2 * obj.bufCapacity);
            end
            row = mod(obj.bufStart + obj.bufCount - 1, obj.bufCapacity) + 1;
            obj.buf(row, :) = x;
            obj.bufCount = obj.bufCount + 1;
            if obj.bufKeep > 0 && obj.bufCount > obj.bufKeep
                obj.dropOldest(obj.bufCount - obj.bufKeep);
            end
        end

        function updateBufferKeepLen(obj)
            obj.bufKeep = IDAT.BUFFER_FACTOR * obj.lambda;
            if obj.bufCount > obj.bufKeep
                obj.dropOldest(obj.bufCount - obj.bufKeep);
            end
        end

        function dropOldest(obj, n)
            obj.bufStart = mod(obj.bufStart - 1 + n, obj.bufCapacity) + 1;
            obj.bufCount = obj.bufCount - n;
        end

        function growBuffer(obj, capacity)
            rows = obj.newestRows(obj.bufCount);
            obj.buf = zeros(capacity, size(obj.buf, 2));
            obj.buf(1:obj.bufCount, :) = flipud(rows);
            obj.bufCapacity = capacity;
            obj.bufStart = 1;
        end

        function W = newestRows(obj, L)
            % The L most recent samples, the newest first.
            rows = mod(obj.bufStart + obj.bufCount - 2 - (0:L-1), obj.bufCapacity) + 1;
            W = obj.buf(rows, :);
        end
    end
end
