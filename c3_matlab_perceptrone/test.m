load iris.dat
x = iris(:,1:4)';
y = iris(:,5);
X = [ones(1,length(x)) ; x]; % bias add
one_hot = diag(ones(1,max(y)));
Y = one_hot(y,:)';
datasample(Y,10)