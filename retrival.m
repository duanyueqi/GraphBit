function retrival()
trn_label = load('train-label.txt');
tst_label = load('test-label.txt');

load('bi_trn_feat.mat');
load('bi_tst_feat.mat');

binary_train = double( bi_trn_feat); 
binary_test = double( bi_tst_feat);
 top_k = 1000; 
 [map, precision_at_k] = precision( trn_label, binary_train', tst_label, binary_test', top_k,1);
fprintf('MAP = %f\n',map);
end
