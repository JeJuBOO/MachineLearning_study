function gradient2 = Gradient(layer,z,gradient1,U1,p_size)
arguments
    layer (:,:,:) double
    z (:,:,:) double
    gradient1 (:,:,:) double
    U1 (:,:,:) double
    p_size (1,1) double
end
[o,p] = size(layer);
gradient2 = zeros(o,p);
[r,q] = size(gradient1);
r_gradient1 = zeros(r,q);
%% CONV-POOL-FC
if q == 1
    for o_idx = 1:o
        for p_idx = 1:p
            D_out_FC = 0;
            for r_idx = 1:r
                D_out_FC = D_out_FC+gradient1(r_idx)* ...
                    U1(r_idx,p_size*fix((o_idx-1)/p_size)+fix((p_idx-1)/p_size)+1);
            end
            gradient2(o_idx,p_idx) = UpSampling(layer,o_idx,p_idx,p_size)* ...
                D_out_FC;%*ReluGradient(z(o_idx,p_idx));
        end
    end
else 
%% CONV-POOL-CONV  
    for o_idx = 1:o
        for p_idx = 1:p
            
            D_out_FC = 0;
            for r_idx = 1:r
                for q_idx = 1:q
                    r_gradient1(r_idx,q_idx) = gradient1(r-r_idx+1,q-q_idx+1);
                end
            end
            [col,row,~,in_n] = size(U1);
            pad = zeros(col+(r-1)*2,row+(q-1)*2,1,in_n);
            pad(r:end-(r-1),q:end-(q-1),:,:) = U1;
            U1_pad = pad;
            for r_idx = 1:r
                for q_idx = 1:q
                    D_out_FC = D_out_FC + ...
                        sum(U1_pad(ceil(o_idx/p_size)+r_idx-1,ceil(p_idx/p_size)+q_idx-1).*r_gradient1,[1,2,3]);
                end
            end
            gradient2(o_idx,p_idx) = UpSampling(layer,o_idx,p_idx,p_size)* ...
                D_out_FC;%*ReluGradient(z(o_idx,p_idx));
            
        end
    end
    
end
end