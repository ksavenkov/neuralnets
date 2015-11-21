function visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states.
    v_in = transpose(rbm_w) * hidden_state;
    num_visible = size(rbm_w,2);
    num_cases = size(hidden_state,2);
    visible_probability = (exp(-1 * v_in) + 1) .^ -1;
end
