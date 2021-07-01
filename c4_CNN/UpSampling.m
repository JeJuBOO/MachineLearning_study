% 21.06.29 sk.boo
% 풀링레이어의 미분
function y = UpSampling(a,o,p,s,method)
arguments
a (:,:) double % 풀링레이어 전의 행렬
o (1,1) double % 인덱스
p (1,1) double
s (1,1) double % 풀링사이즈
method = "max"
end
if method == "max"
    if a(o,p) == max(NB(a,o,p,s))
        y = 1;
    else
        y = 0;
    end
elseif method == "mean"
    y = 1/s*s;
else
    fprintf("method는 Max(defult)또는 mean을 입력하세요")
    
end