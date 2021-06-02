function out = Forward(in,kernel,act_func,stride,padding)
arguments
    in (:,:,:,:) double
    kernel (:,:,:,:) double
    act_func = @Relu
    stride (1,1) double = 1
    padding (1,1) double = 0 %zero padding
end

[col,row,~,in_n] = size(in);
if padding ~= 0
    pad = zeros(col+padding*2,row+padding*2,1,in_n);
    pad(padding+1:end-padding,padding+1:end-padding,:,:) = in;
    in = pad;
end

[col,row,~,in_n] = size(in);
[k_col,k_row,~,k_n] = size(kernel);
j = length(1 :stride: col - k_col+1);
i = length(1 :stride: row - k_row+1);
out = zeros(j,i,1,in_n*k_n);
j_out = 0;i_out = 0;

for j = 1 :stride: col - k_col+1
    j_out = j_out+1;
    for i = 1 :stride: row - k_row+1
        i_out = i_out+1;
        for n = 1:in_n
            out(j_out,i_out,1,k_n*(n-1)+1:n*k_n) = ...
                sum(in(j:k_col-1+j,i:k_row-1+i,:,n).*kernel,[1,2,3]);
        end
    end
    i_out = 0;
end

out = (out-mean(out,[1,2,3]))./std(out,0,[1,2,3]);
out = act_func(out);
end
