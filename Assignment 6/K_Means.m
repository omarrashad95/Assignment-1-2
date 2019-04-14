function [C,miu,miu_ci]=K_Means(X,K)

C=[];
indicies=randi(17999,1,K);
miu=X(:,indicies);
%cluster assignment
min=inf;
indx=0;
for i=1:size(X,2)
    for j=1:K
        distance=sum((X(:,i)-miu(:,j)).^2);
        if(distance<min)
            min=distance;
            indx=j;
        end
    end
    C=[C indx];
end
for i=1:K
    if(length(find(C==i))~=0)
        ind=find(C==i);
        temp=X(:,ind);
        miu(:,i)=sum(temp,2)/length(temp);
    end
end

for i=1:length(C)
    miu_ci(:,i)=miu(:,C(i));
end













end