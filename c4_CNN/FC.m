function out = FC(in,weight)
arguments
    in (:,:,:,:) double
    weight (:,:) double
end
%%
in = reshape(in,[],1);
out = weight*in;
end