clear all;

dim = 16;

load([num2str(dim),'_feat_train.mat']);
load([num2str(dim),'_feat_test.mat']);

train_output = prob_train;
test_output = prob_test;

mean_trn = mean(train_output);
mean_tst = mean(test_output);

for i = 1:dim
    bi_trn_feat(:,i) = (train_output(:,i)>mean_trn(i));
    bi_tst_feat(:,i) = (test_output(:,i)>mean_tst(i));
end


save bi_trn_feat.mat bi_trn_feat
save bi_tst_feat.mat bi_tst_feat
