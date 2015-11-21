function ret = partition(rbm_w)
 [hid,vis] = size(rbm_w)

 % create an array of all possible 2^hid binary masks <2^hid> x <hid>
 masks = bitget(repmat((1:2^hid)',1,hid),repmat(hid:-1:1,2^hid,1));

 % calculate all weight sums per hid configuration per vis unit <2^hid> x <vis>
 w_sums = masks * rbm_w;

 % now exp all elements, add 1, multiply by rows and sum by columns
 E = sum(prod(exp(w_sums) + 1,2));

 ret = log(E);
end
