function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

  %turning real-valued data to binary:
  visible_data = sample_bernoulli(visible_data);

  num_cases = size(visible_data,2);

  %perform CD1

  % get hidden unit probabilities at time 0
  hidden_data_0_prob = visible_state_to_hidden_probabilities(rbm_w,visible_data);
  % sample hidden layer for 10 cases
  hidden_data_0 = sample_bernoulli(hidden_data_0_prob);
  % get visible unitprobabilities at time 1
  visible_data_1_prob = hidden_state_to_visible_probabilities(rbm_w,hidden_data_0);
  % reconstruct visible units at time 1
  visible_data_1 = sample_bernoulli(visible_data_1_prob);
  % get hidden unit probabilities at time 1
  hidden_data_1_prob = visible_state_to_hidden_probabilities(rbm_w,visible_data_1);
  % sample hidden layer at time 1
  hidden_data_1 = sample_bernoulli(hidden_data_1_prob);
  
  %calculate gradients
  goodness_gradient_positive = configuration_goodness_gradient(visible_data, hidden_data_0);
  goodness_gradient_negative = configuration_goodness_gradient(visible_data_1, hidden_data_1);

  ret = goodness_gradient_positive - goodness_gradient_negative;
end
