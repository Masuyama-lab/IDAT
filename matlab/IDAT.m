% Copyright (c) 2025-2026 Naoki Masuyama
% SPDX-License-Identifier: MIT

classdef IDAT < handle
    % IDAT (Inverse Distance ART with Topology)
    % Implements an ART-based topological clustering algorithm (IDAT) with automatic adjustment of the vigilance parameter.
    %
    % Public Methods:
    %   - fit(samples): Performs sequential clustering training (in-place; modifies the model).
    %   - predict(samples): Performs sequential clustering testing (read-only; uses current topology/weights).
    %
    % Private Static Methods:
    %   - compute_inverse_distance_matrix(...): Computes pairwise Euclidean distance matrix and its inverse-distance similarity matrix s=1/(1+α d).
    %
    % Private Methods:
    %   - cluster_step(...): Clusters an individual sample, updates prototypes, manages edges, and periodically prunes long edges and underused nodes.
    %   - calculate_lambda_similarity_threshold(...): Computes the Lambda interval and similarity threshold from buffered statistics using stability checks.
    %   - calculate_lambda_decremental_direction(...): Reduces Lambda when inverse-distance stability degrades (search from small to large; stop at first instability).
    %   - calculate_lambda_incremental_direction(...): Attempts to increase Lambda using only past data (causal, no lookahead) up to 2*Lambda via exponential search.
    %   - candidate_similarity_threshold(...): Computes a candidate similarity threshold from row-wise maxima and the cluster-to-node ratio with exponential smoothing.
    %   - update_online_variance(...), reset_online_variance(...): Maintain a global online standard deviation estimate via Welford's algorithm (numeric stability).
    %   - add_cluster_node(...): Adds a new node with initialization.
    %   - append_to_buffer(...), get_from_buffer(...), clear_buffer(...), update_buffer_keep_len(...): Ring-buffer utilities with adaptive capacity and logical retention.
    %
    % Public Properties:
    %   - numNodes:            Current number of prototype nodes (K).
    %   - weight:              [K x d] prototype matrix; each row is a node prototype.
    %   - CountNode:           [K x 1] usage counts; governs learning rates (1/t schedule).
    %   - bufferStd:           [K x 1] per-node values updated from global variance (used to form alpha for similarities; alpha_k=1/bufferStd_k).
    %   - buffer_data:         Ring buffer storing recent samples (oldest → newest). Capacity can exceed logical retention for amortized O(1) appends.
    %   - edge:                [K x K] sparse adjacency; >0 indicates an existing edge (undirected, maintained symmetrically).
    %   - Lambda:              Window length for stability evaluation and buffer policy (logical retention length is 2*Lambda).
    %   - similarityTh:        Vigilance threshold in [0, 1). Match requires sim ≥ similarityTh.
    %   - recalcCounter:       Per-sample counter to trigger (Lambda, similarityTh) updates when it reaches Lambda.
    %   - edgeCandidateCounter:[K x K] co-activation counters used to form edges (thresholded by running mean of positive entries).
    %   - isWeight:            [K x 1] logical flags; if any are true, only those nodes are used in prediction; otherwise all nodes are used (fallback).
    %   - LabelCluster:        [1 x K] connected-component labels over the current graph (computed in predict).
    %   - sample_counts:       Total number of processed samples.
    % 

    
    properties (Access = public)
        numNodes
        weight
        CountNode
        bufferStd
        buffer_data
        edge
        Lambda
        similarityTh
        recalcCounter
        edgeCandidateCounter
        isWeight
        LabelCluster
        sample_counts
    end
    
    properties (Access = private)
        % Online variance (Welford)
        % meanDataForNode, M2DataForNode, countDataForNode are global (not per-node); used to estimate a robust scale for alpha computation.
        meanDataForNode
        M2DataForNode
        countDataForNode
        
        % Ring buffer internals
        % buffer_data is a circular array with logical head at buffer_data_start and count buffer_data_count.
        buffer_data_capacity
        buffer_data_start
        buffer_data_count

        % Logical retention length (2*Lambda)
        % We retain at most buffer_keep_len recent samples logically, while capacity may be larger to avoid frequent reallocations.
        buffer_keep_len
        
        % Smoothed cluster-to-node ratio
        % Exponentially smoothed ratioNC := (#clusters / #nodes), mixing rate 1/Lambda; used to pick quantile for similarityTh.
        past_ratioNC

        % Demotion state (tolerance-based)
        % Snapshots and counters to decide demotion of isWeight=true nodes when isolated and inactive for several maintenance intervals.
        prev_CountNode
        prev_edgeRowSum
        demotionToleranceCount
    end
    
    methods (Access = public)
        %% Constructor
        function obj = IDAT()
            % Initialize all public properties to their default values.
            obj.numNodes             = 0;
            obj.weight               = [];
            obj.CountNode            = [];
            obj.bufferStd            = [];
            obj.buffer_data          = [];
            obj.edge                 = sparse(2,2);
            obj.Lambda               = 2;
            obj.similarityTh         = 0;
            obj.recalcCounter        = 0;
            obj.edgeCandidateCounter = sparse(2,2);
            obj.isWeight             = [];
            obj.LabelCluster         = [];
            obj.meanDataForNode      = [];
            obj.M2DataForNode        = [];
            obj.countDataForNode     = 0;
            obj.buffer_data_capacity = 0;
            obj.buffer_data_start    = 1;
            obj.buffer_data_count    = 0;
            obj.buffer_keep_len      = 0;

            obj.past_ratioNC         = 0;
            obj.sample_counts        = 0;

            % Initialize demotion state
            % CHANGE: initialize as 0-by-1 column vectors to prevent accidental row expansion.
            obj.prev_CountNode       = 0;
            obj.prev_edgeRowSum      = 0;
            obj.demotionToleranceCount  = 0;
        end
        
        %% Fit Method
        function obj = fit(obj, samples)
            % Convert input to double for consistent math.
            samples = double(samples);
            
            % Process sequentially (no lookahead used in learning).
            for sampleNum = 1:size(samples,1)

                sample = samples(sampleNum, :);
                
                % Learn current sample
                obj.append_to_buffer(sample);
                obj.cluster_step(sample);

                obj.sample_counts = obj.sample_counts + 1;
                obj.recalcCounter = obj.recalcCounter + 1;

                if obj.recalcCounter >= obj.Lambda && size(obj.weight,1) > 2
                    % Current cluster count from the graph
                    % Complexity: building a graph over K nodes; conncomp is near O(K+E) for sparse edge.
                    connection  = graph(obj.edge ~= 0);
                    numClusters = max(conncomp(connection));

                    % Decremental direction first (uses only past Lambda samples)
                    % Rationale: reduce Lambda if instability appears earlier
                    pastLambdaBuf = obj.get_from_buffer(obj.Lambda);
                    [Lambda_new, similarityTh_new, isIncremental, ratioNC_candidate] = obj.calculate_lambda_decremental_direction(obj.bufferStd, obj.Lambda, obj.similarityTh, pastLambdaBuf, obj.numNodes, numClusters);

                    % Incremental direction (NO lookahead): widen using older past samples up to 2*Lambda
                    % We only attempt widening if decremental scan did not detect instability (isIncremental==true).
                    if isIncremental
                        [Lambda_new, similarityTh_new, ratioNC_candidate] = obj.calculate_lambda_incremental_direction(obj.bufferStd, obj.Lambda, obj.similarityTh, obj.numNodes, numClusters);
                    end

                    % Commit updates and reset counter
                    obj.Lambda        = Lambda_new;
                    obj.similarityTh  = similarityTh_new;
                    obj.past_ratioNC  = ratioNC_candidate;
                    obj.recalcCounter = 0;

                    % Recompute retention and proactively grow capacity
                    obj.update_buffer_keep_len();
                end

                
            end
        end

        %% Predict Method
        function predicted_labels = predict(obj, samples)
            % Prediction uses current graph components as cluster labels.
            % If some nodes are marked in isWeight, restrict winners to that subset; otherwise use all nodes.
            % Numerics: alpha_k = 1 / bufferStd_k with guard to avoid division by zero.
            % Output: predicted_labels is a row vector of length N (samples).
            if isempty(obj.isWeight) || ~any(obj.isWeight)
                weightIndices = 1:obj.numNodes;
            else
                weightIndices = find(obj.isWeight);
            end
        
            % Create graph from edges and compute connected components (cluster labels)
            connection = graph(obj.edge ~= 0);
            obj.LabelCluster = conncomp(connection);
        
            % Convert samples to double
            samples = double(samples);

            % Guard for empty model or mask
            if isempty(weightIndices) || obj.numNodes == 0
                predicted_labels = [];
                return;
            end

            % Node-wise alpha from bufferStd
            if isempty(obj.bufferStd)
                alpha_vec = ones(numel(weightIndices), 1);
            else
                bs = obj.bufferStd(weightIndices, 1);
                bs(~isfinite(bs) | bs <= 0) = 1.0e-6;
                alpha_vec = 1 ./ bs;
            end

            % Compute Euclidean distances from nodes to samples: [K' x N]
            D2 = pdist2(obj.weight(weightIndices,:), samples, 'squaredeuclidean');
            D  = sqrt(max(D2, 0));

            % Similarity with node-wise alpha: s = 1 / (1 + alpha_k * d)
            S = 1 ./ (1 + bsxfun(@times, alpha_vec, D));

            % Winner per sample = argmax similarity
            [~, winnerIdx] = max(S, [], 1);

            % Assign predicted labels
            predicted_labels = obj.LabelCluster(weightIndices(winnerIdx));
        end
        
        %% Clustering Step for a Single Sample
        function obj = cluster_step(obj, sample)
            % Per-sample update:
            %   - If K<3, seed nodes quickly to establish a baseline scale (bufferStd mirrored for stability).
            %   - Else, perform resonance test with vigilance, update winner(s), and manage edge candidates.
            %   - Periodically, prune overly long edges via IQR and remove rarely used nodes to keep the graph compact.
            if obj.numNodes < 3
                obj.add_cluster_node(sample);
                if obj.numNodes == 2
                    obj.bufferStd(1) = obj.bufferStd(2);
                end
            else
                % Category Choice
                distances = sqrt(sum((obj.weight - sample).^2, 2));
                [~, sortedIdx] = sort(distances, 'ascend');
                alpha = 1 ./ obj.bufferStd(:, 1);
                allSims = 1 ./ (1 + alpha .* distances);
                baselineVigilance = obj.similarityTh;

                % Direct two-nearest decision without while-loop
                % s1 is the nearest, s2 is the second nearest (by distance order).
                s1 = sortedIdx(1);
                s2 = sortedIdx(2);
                sim_s1 = allSims(s1);
                sim_s2 = allSims(s2);

                if sim_s1 < baselineVigilance
                    % No node satisfies vigilance: create a new node immediately.
                    obj.add_cluster_node(sample);
                else
                    % Update s1 node (prototype and usage).
                    obj.CountNode(s1) = obj.CountNode(s1) + 1;
                    obj.weight(s1,:)  = obj.weight(s1,:) + (1 / obj.CountNode(s1)) * (sample - obj.weight(s1,:));
                    [~, currentStd]   = obj.update_online_variance(sample);
                    obj.bufferStd(s1, 1) = max(max(currentStd), 1.0e-6);

                    if sim_s2 > baselineVigilance
                        % Update s2 node (prototype and usage).
                        obj.CountNode(s2) = obj.CountNode(s2) + 1;
                        obj.weight(s2,:) = obj.weight(s2,:) + (1 / (obj.CountNode(s1) + obj.CountNode(s2))) * (sample - obj.weight(s2,:));
                        % Use the mean of valid counts as robustness threshold to avoid very young nodes.
                        validCountNode = obj.CountNode(obj.CountNode > 1);
                        CountNodeThreshold = mean(validCountNode);

                        if obj.CountNode(s1) > CountNodeThreshold && obj.CountNode(s2) > CountNodeThreshold
                            % Mark both as candidates for weighted prediction and accumulate co-activations.
                            obj.isWeight(s1) = true;
                            obj.isWeight(s2) = true;
                            obj.edgeCandidateCounter(s1, s2) = obj.edgeCandidateCounter(s1, s2) + 1;
                            obj.edgeCandidateCounter(s2, s1) = obj.edgeCandidateCounter(s2, s1) + 1;
                            
                            % Edge creation threshold: running mean of positive counters.
                            valsOver = nonzeros(obj.edgeCandidateCounter);
                            edgeThreshold = mean(valsOver);
                            
                            if obj.edgeCandidateCounter(s1, s2) > edgeThreshold
                                % Define an edge between s1 and s2 node.
                                obj.edge(s1, s2) = obj.edge(s1, s2) + 1;
                                obj.edge(s2, s1) = obj.edge(s2, s1) + 1;
                            end
                        end
                    end
                end
            end
            
            % Periodic maintenance
            % Every Lambda samples:
            %   1) Remove overly long edges by Tukey's rule (IQR) on edge lengths in prototype space.
            %   2) Remove some nodes with CountNode==1 if they accumulate beyond Lambda (conservative pruning).
            %   3) Demote isolated and inactive isWeight nodes via tolerance counting to avoid overfitting.
            if mod(obj.sample_counts, obj.Lambda) == 0
                [rowIdx, colIdx] = find(obj.edge > 0);
                if ~isempty(rowIdx)
                    edgeLengths = sqrt(sum((obj.weight(rowIdx,:) - obj.weight(colIdx,:)).^2, 2));
                    if ~isempty(edgeLengths)
                        Q1 = quantile(edgeLengths, 0.25);
                        Q3 = quantile(edgeLengths, 0.75);
                        IQR = Q3 - Q1;
                        upper_bound = Q3 + 1.5 * IQR;
                        tooLongIdx = (edgeLengths > upper_bound);
                        linIdx = sub2ind(size(obj.edge), rowIdx(tooLongIdx), colIdx(tooLongIdx));
                        obj.edge(linIdx) = 0;
                        obj.edge(sub2ind(size(obj.edge), colIdx(tooLongIdx), rowIdx(tooLongIdx))) = 0;

                        % Reset edgeCandidateCounter to avoid immediate re-creation without fresh evidence.
                        obj.edgeCandidateCounter(linIdx) = 0;
                        obj.edgeCandidateCounter(sub2ind(size(obj.edgeCandidateCounter), colIdx(tooLongIdx), rowIdx(tooLongIdx))) = 0;
                    end
                end

                % Conservative removal of nodes with CountNode==1 when they exceed Lambda in number.
                candidatesToRemove = find(obj.CountNode == 1);
                if ~all(obj.CountNode == 1)
                    if ~isempty(candidatesToRemove)
                        if length(candidatesToRemove) > obj.Lambda
                            nodesToRemove = candidatesToRemove(1:end - obj.Lambda);
                        else
                            nodesToRemove = [];
                        end
                        if ~isempty(nodesToRemove)
                            % Remove rows/cols consistently across all node-indexed structures; update obj.numNodes afterwards.
                            obj.edge(nodesToRemove, :) = [];
                            obj.edge(:, nodesToRemove) = [];
                            obj.weight(nodesToRemove, :) = [];
                            obj.CountNode(nodesToRemove) = [];
                            obj.bufferStd(nodesToRemove, :) = [];
                            obj.isWeight(nodesToRemove) = [];
                            obj.edgeCandidateCounter(nodesToRemove, :) = [];
                            obj.edgeCandidateCounter(:, nodesToRemove) = [];
                            obj.numNodes = size(obj.weight, 1);
                            
                            obj.prev_CountNode(nodesToRemove) = [];
                            obj.prev_edgeRowSum(nodesToRemove) = [];
                            obj.demotionToleranceCount(nodesToRemove) = [];
                        end
                    end
                end

                % Demote isolated and low-activity nodes (isWeight = true => false)
                obj = obj.demote_isweight();
            end
        end
        
        %% Lambda and Similarity Threshold Calculation
        function [Lambda_new, similarityTh_new, isRenewed, ratioNC_local_out] = calculate_lambda_similarity_threshold(obj, bufferStd, buffer, currentIdx, data_size, numNodes, numClusters, finalizeOnExhaustion, suppressUpdate, ratioNC_local_in)
            % Decide new Lambda and similarity threshold from the buffered data.
            % Inputs:
            %   bufferStd: scalar scale to define alpha=1/max(bufferStd).
            %   buffer:    chronological samples used for stability assessment (oldest→newest).
            %   currentIdx, data_size: cursor for scanning; if finalizeOnExhaustion, renew at end.
            %   numNodes, numClusters: topology stats; may be empty during bootstrapping.
            %   finalizeOnExhaustion: if true, renewal is allowed at window end even if stable.
            %   suppressUpdate: if true, do not write back to obj.past_ratioNC (local what-if).
            %   ratioNC_local_in: local ratioNC state for pure functions in search.
            % Outputs:
            %   Lambda_new, similarityTh_new: proposed window and vigilance.
            %   isRenewed: true iff a renewal decision is made at currentIdx.
            %   ratioNC_local_out: local ratioNC after mixing (may be committed by caller).
            %
            % Stability criterion:
            %   Build inverse-distance similarity matrix S=1/(1+α D). If S is near-singular, deem it unstable.
            %   Use Cholesky factorization only; if it fails (non-PD), mark as unstable. If it succeeds, use prod(diag(L))^2 threshold.
            if nargin < 8 || isempty(finalizeOnExhaustion)
                finalizeOnExhaustion = true;
            end
            if nargin < 9 || isempty(suppressUpdate)
                suppressUpdate = false;
            end
            if nargin < 10 || isempty(ratioNC_local_in)
                ratioNC_local_in = obj.past_ratioNC;
            end
            Lambda_new = [];
            similarityTh_new = [];
            isRenewed = false;
            ratioNC_local_out = ratioNC_local_in;

            alpha = 1 / max(bufferStd);
            [~, invDistMat] = obj.compute_inverse_distance_matrix(buffer, alpha);

            % Cholesky-based stability check
            [L, p] = chol(invDistMat, 'lower');
            isUnstable = false;
            if p == 0
                % Use Cholesky factor to approximate determinant (det = prod(diag(L))^2)
                detApprox = prod(diag(L))^2;
                if detApprox < 1.0e-6
                    isUnstable = true;
                end
            else
                % Non-PD: declare instability without determinant fallback
                isUnstable = true;
            end
            
            % Finalize if unstable or (optionally) if the candidate window is exhausted.
            if isUnstable || (finalizeOnExhaustion && currentIdx == data_size)
                Lambda_new = size(buffer, 1);
                % For thresholding, ignore trivial self-similarity on diagonal.
                invDistMat(1:size(buffer, 1)+1:end) = -inf; % remove diagonal
                maxValues = max(invDistMat, [], 2);
                if isempty(numNodes)
                    % During bootstrap, fallback to mean of maxima as a coarse threshold.
                    similarityTh_new = mean(maxValues);
                    % ratioNC_local_out unchanged when clusters/nodes are unavailable
                else
                    [similarityTh_new, ratioNC_local_out] = obj.candidate_similarity_threshold(numClusters, numNodes, maxValues, obj.Lambda, false, ratioNC_local_in);
                end
                isRenewed = true;
            end
        end
        
    end
    
    methods (Access = private)

        %% Online Standard Deviation Update Method (Welford)
        function [obj, currentStd] = update_online_variance(obj, newSamples)
            % Update Welford's running variance for global feature-wise scale estimation.
            % Numerically stable for streaming; O(n*d) per call for n new rows and d features.
            numNewSamples = size(newSamples, 1);
            if isempty(obj.meanDataForNode)
                obj.meanDataForNode = mean(newSamples, 1);
                obj.M2DataForNode   = zeros(size(obj.meanDataForNode));
                obj.countDataForNode = numNewSamples;
                currentStd = zeros(size(obj.meanDataForNode));
            else
                newCount = obj.countDataForNode + numNewSamples;
                delta    = newSamples - obj.meanDataForNode;
                obj.meanDataForNode = obj.meanDataForNode + sum(delta, 1) / newCount;
                obj.M2DataForNode   = obj.M2DataForNode + sum(delta .* (newSamples - obj.meanDataForNode), 1);
                obj.countDataForNode = newCount;
                variance = obj.M2DataForNode / (obj.countDataForNode - 1);
                variance(variance < 0) = 0;
                currentStd = sqrt(variance);
            end
        end

        %% Reset Online Standard Deviation Variables
        function reset_online_variance(obj)
            % Reset streaming statistics; used during hyperparameter re-initialization searches.
            obj.meanDataForNode = [];
            obj.M2DataForNode   = [];
            obj.countDataForNode = 0;
        end
        
        function [Lambda_new, similarityTh_new, isIncrease, ratioNC_local] = calculate_lambda_decremental_direction(obj, bufferStd, Lambda, similarityTh, buffer, numNodes, numClusters)
            % Analyze stability and try to reduce Lambda if unstable.
            % Search strategy:
            %   - Reverse the buffer (most recent first), factor once by Cholesky, and test growing principal minors.
            %   - If full-matrix Cholesky fails, test each prefix by Cholesky (no determinant fallback).
            %   - Stop at the first minor whose approximate determinant < 1e-6 or whose Cholesky fails; return last valid window.
            isIncrease = true;
            lastValidLambda = Lambda;
            lastValidsimilarityTh = similarityTh;

            % Initialize local copy of ratioNC from the object's state
            ratioNC_local = obj.past_ratioNC;

            alpha = 1 / max(bufferStd);
            [fullDistMat, invDistMatFull] = obj.compute_inverse_distance_matrix(flipud(buffer), alpha);
            [L, p] = chol(invDistMatFull, 'lower');
            startN = 2;
            endN = size(buffer, 1);
            
            if p == 0
                for currentN = startN:endN
                    subDetValue = prod(diag(L(1:currentN, 1:currentN)))^2;
                    if subDetValue < 1.0e-6
                        Lambda_new = lastValidLambda;
                        similarityTh_new = lastValidsimilarityTh;
                        isIncrease = false;
                        return;
                    end
                    % Renewal candidate at currentN
                    lastValidLambda = currentN;
                    subMat = invDistMatFull(1:currentN, 1:currentN);
                    subMat(1:currentN+1:end) = -inf;
                    maxValues = max(subMat, [], 2);
                    [lastValidsimilarityTh, ratioNC_local] = obj.candidate_similarity_threshold(numClusters, numNodes, maxValues, obj.Lambda, true, ratioNC_local);
                end
            else
                % Full Cholesky failed: test each prefix using Cholesky only
                for currentN = startN:endN
                    subDistMat = fullDistMat(1:currentN, 1:currentN);
                    invDistMat = 1 ./ (1 + alpha * subDistMat);
                    [Lsub, psub] = chol(invDistMat, 'lower');
                    if psub ~= 0
                        % Non-PD prefix: declare instability at currentN
                        Lambda_new = lastValidLambda;
                        similarityTh_new = lastValidsimilarityTh;
                        isIncrease = false;
                        return;
                    end
                    % PD prefix: use Cholesky-based approximate determinant
                    subDetValue = prod(diag(Lsub))^2;
                    if subDetValue < 1.0e-6
                        Lambda_new = lastValidLambda;
                        similarityTh_new = lastValidsimilarityTh;
                        isIncrease = false;
                        return;
                    end
                    lastValidLambda = currentN;
                    invDistMat(1:currentN+1:end) = -inf;
                    maxValues = max(invDistMat, [], 2);
                    [lastValidsimilarityTh, ratioNC_local] = obj.candidate_similarity_threshold(numClusters, numNodes, maxValues, obj.Lambda, true, ratioNC_local);
                end
            end
            Lambda_new = lastValidLambda;
            similarityTh_new = lastValidsimilarityTh;
        end
        
        function [Lambda_new, similarityTh_new, ratioNC_local] = calculate_lambda_incremental_direction(obj, bufferStd, Lambda, similarityTh, numNodes, numClusters)
            % Attempt to widen Lambda up to 2*Lambda using only older past data (no future lookahead).
            % Strategy:
            %   - Determine available past samples in buffer; cap at 2*Lambda.
            %   - Exponential search to find the first K that triggers renewal; then set Lambda_new to the previous stable lowK.
            %   - If no trigger until cap, finalize at cap and compute threshold.
            % Outputs: proposed Lambda_new and vigilance, plus local ratioNC to be committed by caller.
        
            Lambda_new = Lambda;
            similarityTh_new = similarityTh;

            % Initialize local copy of ratioNC from the object's state
            ratioNC_local = obj.past_ratioNC;
        
            % How many past samples we can use now
            availPast = obj.getBufferDataCount();
            if availPast <= Lambda
                % Nothing older than the last Lambda samples; cannot increase.
                return;
            end
        
            endIndex = min(2*Lambda, availPast);
        
            % ---- Exponential search to find the first triggering k ----
            step = 1;
            lowK = Lambda; % known "not triggered" side
            highK = min(Lambda + step, endIndex);
            found = false;
        
            while highK <= endIndex
                extend_buffer = obj.get_from_buffer(highK);
                [~, ~, isRenewed, ratioNC_local] = obj.calculate_lambda_similarity_threshold( ...
                    bufferStd, extend_buffer, highK, endIndex, numNodes, numClusters, false, true, ratioNC_local);
        
                if isRenewed
                    found = true;
                    break; % [lowK, highK] now brackets the first trigger
                else
                    lowK = highK;
                    % grow step exponentially but do not pass the cap
                    step = min(step * 2, endIndex - Lambda);
                    highK = min(Lambda + step, endIndex);
                    if highK <= lowK
                        break; % safety
                    end
                end
            end
        
            if found
                % Choose the largest stable window (lowK) just before the first trigger.
                Lambda_new = lowK;
        
                % Recompute similarity threshold on a buffer of length lowK
                stable_buffer = obj.get_from_buffer(lowK);
                alpha = 1 / max(bufferStd);
                [~, invDistMat] = obj.compute_inverse_distance_matrix(stable_buffer, alpha);
                invDistMat(1:lowK+1:end) = -inf;
                maxValues = max(invDistMat, [], 2);
                if isempty(numNodes)
                    similarityTh_new = mean(maxValues);
                else
                    [similarityTh_new, ratioNC_local] = obj.candidate_similarity_threshold(numClusters, numNodes, maxValues, lowK, true, ratioNC_local);
                end
                return;
            else
                % No trigger; finalize at cap with full stability check and thresholding.
                cap_buffer = obj.get_from_buffer(endIndex);
                [Lambda_new, similarityTh_new, ~, ratioNC_local] = obj.calculate_lambda_similarity_threshold( ...
                    bufferStd, cap_buffer, endIndex, endIndex, numNodes, numClusters, true, true, ratioNC_local);
            end
        end

        
        %% Ring Buffer Operations
        function append_to_buffer(obj, newData)
            % Append new data samples to the ring buffer.
            % Capacity policy: expand with headroom only; never shrink (amortized O(1) appends).
            % Logical retention: keep only the most recent buffer_keep_len samples; capacity may be larger.
            [numNew, numCols] = size(newData);
            if obj.buffer_data_count == 0
                % Initial capacity with headroom against current keep_len
                initialCapacity = max([500, numNew, ceil(2* max(1, obj.buffer_keep_len))]);
                obj.buffer_data_capacity = initialCapacity;
                obj.buffer_data = zeros(initialCapacity, numCols);
                obj.buffer_data_start = 1;
            end
            
            requiredCapacity = obj.buffer_data_count + numNew;
            if requiredCapacity > obj.buffer_data_capacity
                % Expand to the larger of:
                % - double current capacity
                % - required capacity times 2
                % - reserve_factor * keep_len (here 2*keep_len)
                newCapacity = max([ ...
                    obj.buffer_data_capacity * 2, ...
                    requiredCapacity * 2, ...
                    ceil(2 * max(1, obj.buffer_keep_len)) ...
                ]);
                newBuffer = zeros(newCapacity, numCols);
                currentData = obj.get_from_buffer();
                currentCount = size(currentData,1);
                if currentCount > 0
                    newBuffer(1:currentCount, :) = currentData;
                end
                obj.buffer_data = newBuffer;
                obj.buffer_data_capacity = newCapacity;
                obj.buffer_data_start = 1;
                obj.buffer_data_count = currentCount;
            end
            
            % Write newData at the logical tail, possibly wrapping around.
            indices = mod((obj.buffer_data_start + obj.buffer_data_count - 1) + (0:numNew-1), obj.buffer_data_capacity) + 1;
            obj.buffer_data(indices, :) = newData;
            obj.buffer_data_count = obj.buffer_data_count + numNew;

            % Logical trimming to keep_len (do not shrink capacity here)
            keepLen = obj.buffer_keep_len;
            if isempty(keepLen) || keepLen <= 0 || ~isfinite(keepLen)
                keepLen = obj.buffer_data_count;
            end
            if obj.buffer_data_count > keepLen
                excess = obj.buffer_data_count - keepLen;
                obj.buffer_data_start = mod(obj.buffer_data_start - 1 + excess, obj.buffer_data_capacity) + 1;
                obj.buffer_data_count = obj.buffer_data_count - excess;
            end
        end
        
        function data = get_from_buffer(obj, reqNum)
            % Retrieve the most recent reqNum elements in chronological order (oldest → newest).
            % If reqNum is omitted, return the entire logical content.
            if nargin < 2 || isempty(reqNum)
                reqNum = obj.buffer_data_count;
            end
            if reqNum > obj.buffer_data_count
                reqNum = obj.buffer_data_count;
            end
            if reqNum == 0
                data = [];
                return;
            end
            
            startIndex = mod(obj.buffer_data_start + obj.buffer_data_count - reqNum - 1, obj.buffer_data_capacity) + 1;
            if startIndex + reqNum - 1 <= obj.buffer_data_capacity
                data = obj.buffer_data(startIndex:startIndex+reqNum-1, :);
            else
                firstPart  = obj.buffer_data(startIndex:obj.buffer_data_capacity, :);
                secondPart = obj.buffer_data(1:(startIndex+reqNum-1-obj.buffer_data_capacity), :);
                data = [firstPart; secondPart];
            end
        end
        
        function count = getBufferDataCount(obj)
            % Return logical number of stored samples (≤ capacity).
            count = obj.buffer_data_count;
        end
        
        function clear_buffer(obj)
            % Clear logical content but keep capacity (for performance).
            obj.buffer_data_count = 0;
            obj.buffer_data_start = 1;
        end

        % Recompute ring-buffer retention length after Lambda update
        function update_buffer_keep_len(obj)
            % Logical retention length = 2*Lambda (at least Lambda).
            % After updating buffer_keep_len, trim logical content and optionally grow capacity with headroom.
            if isempty(obj.Lambda)
                obj.buffer_keep_len = obj.buffer_data_count; % nothing to do yet
                return;
            end
            keepLen = max(obj.Lambda, 2 * obj.Lambda);
            obj.buffer_keep_len = max(keepLen, obj.Lambda); % effectively 2*Lambda

            % Logical trim if current stored count exceeds keep_len
            if obj.buffer_data_count > obj.buffer_keep_len
                excess = obj.buffer_data_count - obj.buffer_keep_len;
                obj.buffer_data_start = mod(obj.buffer_data_start - 1 + excess, obj.buffer_data_capacity) + 1;
                obj.buffer_data_count = obj.buffer_data_count - excess;
            end

            % Proactive capacity growth to reserve headroom (no shrink)
            targetCapacity = ceil(2 * obj.buffer_keep_len);
            if obj.buffer_data_capacity < targetCapacity
                numCols = size(obj.buffer_data, 2);
                if numCols == 0
                    % No buffer allocated yet; delay allocation until first append.
                    obj.buffer_data_capacity = targetCapacity;
                    return;
                end
                newBuffer = zeros(targetCapacity, numCols);
                currentData = obj.get_from_buffer();
                currentCount = size(currentData,1);
                if currentCount > 0
                    newBuffer(1:currentCount, :) = currentData;
                end
                obj.buffer_data = newBuffer;
                obj.buffer_data_capacity = targetCapacity;
                obj.buffer_data_start = 1;
                obj.buffer_data_count = currentCount;
            end
        end
        
        %% Candidate Similarity Threshold Calculation
        function [new_threshold, ratioNC_out] = candidate_similarity_threshold(obj, numClusters, numNodes, maxValues, Lambda, suppressUpdate, ratioNC_in)
            % Compute threshold from row-wise maxima of invDistMat and cluster/node ratio.
            % Let ratioNC := #clusters/#nodes. We smooth it: ratioNC ← (1/Lambda)*current + (1-1/Lambda)*ratioNC.
            % Then choose quantile q = 1 - ratioNC and set threshold as quantile(maxValues, q).
            % Intuition: More clusters per node (higher ratio) => lower q => more conservative vigilance (fewer matches).
            % Side-effects:
            %   - If suppressUpdate=true, obj.past_ratioNC is not updated (used inside search procedures).
            %   - Returns ratioNC_out to allow caller-controlled commit.
            if nargin < 6 || isempty(suppressUpdate)
                suppressUpdate = false;
            end
            if nargin < 7 || isempty(ratioNC_in)
                ratioNC_in = obj.past_ratioNC;
            end

            current_ratioNC = numClusters / numNodes;
            mixing_ratio     = 1 / Lambda;
            mixed_ratioNC    = mixing_ratio * current_ratioNC + (1 - mixing_ratio) * ratioNC_in;

            % Commit to the object only when not suppressed
            if ~suppressUpdate
                obj.past_ratioNC = mixed_ratioNC;
            end

            qValue = (1 - mixed_ratioNC); % q in [0,1]
            new_threshold = quantile(maxValues, qValue);
            ratioNC_out   = mixed_ratioNC;
        end
        
        %% Node Addition Function
        function add_cluster_node(obj, sample)
            % Create a new node initialized at the current sample; mark unweighted until it gains sufficient evidence.
            obj.numNodes = obj.numNodes + 1;
            idx = obj.numNodes;
            obj.weight(idx,:) = sample;
            obj.edge(idx,:) = 0;
            obj.edge(:,idx) = 0;
            obj.CountNode(idx) = 1;
            obj.isWeight(idx) = false;
            obj.edgeCandidateCounter(idx,:) = 0;
            obj.edgeCandidateCounter(:,idx) = 0;
            
            [~, currentStd] = obj.update_online_variance(sample);
            obj.bufferStd(idx,1) = max(max(currentStd), 1.0e-6);

            obj.prev_CountNode(idx) = 0;
            obj.prev_edgeRowSum(idx) = 0;
            obj.demotionToleranceCount(idx) = 0;
        end

        %% Demotion of isWeight=true based on internal activity statistics
        function obj = demote_isweight(obj)
            % Demote isWeight=true nodes if they remain isolated (degree==0) and
            % comparatively inactive; track tolerance with demotionToleranceCount and
            % trigger demotion when it reaches demotionToleranceThreshold ≈ max(1, round(buffer_keep_len/Lambda)).

            % Degree per node
            degree = full(sum(obj.edge > 0, 2));  % Kx1

            % Window increments since last maintenance
            usageCount = obj.CountNode(:);
            deltaUsage = usageCount - obj.prev_CountNode;

            coactRowSum = full(sum(obj.edgeCandidateCounter, 2));
            deltaCoact  = coactRowSum - obj.prev_edgeRowSum;

            % Activity score for this maintenance interval
            activityScore = deltaUsage + deltaCoact;

            % If no positive activity observed, reset tolerance and take snapshots
            hasActivity = (activityScore > 0);
            if ~any(hasActivity)
                obj.prev_CountNode  = usageCount;
                obj.prev_edgeRowSum = coactRowSum;
                obj.demotionToleranceCount(:) = 0;
                return;
            end

            % Robust lower bound (Tukey's rule) on positive activities
            activityVals = activityScore(hasActivity);
            Q1 = quantile(activityVals, 0.25);
            Q3 = quantile(activityVals, 0.75);
            IQR = Q3 - Q1;
            lowerBound = max(Q1 - 1.5 * IQR, 0);  % never below zero

            % Candidates: isWeight & degree==0 & low activity (<= lowerBound)
            isWeighted = obj.isWeight(:);
            isCandidate = logical(isWeighted) & (degree == 0) & (activityScore <= lowerBound);

            % Update tolerance counts; reset when condition not met
            obj.demotionToleranceCount(isCandidate)  = obj.demotionToleranceCount(isCandidate) + 1;
            obj.demotionToleranceCount(~isCandidate) = 0;

            % Compute demotionToleranceThreshold from internal policy
            % demotionToleranceThreshold = max(1, round(max(1, obj.buffer_keep_len) / max(1, obj.Lambda)));
            demotionToleranceThreshold = 2;

            % Demote when tolerance count reaches threshold; then reset
            demoteIdx = find(obj.demotionToleranceCount >= demotionToleranceThreshold);
            obj.isWeight(demoteIdx) = false;
            obj.demotionToleranceCount(demoteIdx) = 0;

            % Roll snapshots forward
            obj.prev_CountNode  = usageCount;
            obj.prev_edgeRowSum = coactRowSum;
        end
    end
    
    methods (Static, Access = private)
        %% Compute Inverse Distance Matrix
        function [distMat, invDistMat] = compute_inverse_distance_matrix(X, alpha)
            % Compute pairwise Euclidean distances and inverse-distance similarities.
            % Inputs: X [n x d], alpha>0 (scale). Output:
            %   distMat[i,j] = ||x_i - x_j||_2
            %   invDistMat[i,j] = 1 / (1 + alpha * distMat[i,j])
            % Numerics: clamp small negatives before sqrt to 0; O(n^2 d) time, O(n^2) memory.
            XX = sum(X.^2, 2);
            D_sq = bsxfun(@plus, XX, XX') - 2 * (X * X');
            distMat = sqrt(max(D_sq, 0));
            invDistMat = 1 ./ (1 + alpha * distMat);
        end
    end
end
