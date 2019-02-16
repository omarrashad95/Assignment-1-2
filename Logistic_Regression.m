function [J,theta]=Logistic_Regression(H,Y,X,theta)
l=0.004;
alpha=0.001; %learning rate
c=1;
J(c)=1/(2*length(H))*sum(-Y'*log(H')-(1-Y')*log(1-H'))+l/(2*length(H))*sum(theta(2:length(theta)).^2);%formulation of the MSE
stop=false;
%gradient descent
while stop==false
    for k=1:length(theta)
        theta(k)=theta(k)*(1-((alpha*l)/length(H)))-(alpha/length(H))*((H-Y')*X(k,:)');
    end
    H=1./(1+exp(-(theta*X)));
    c=c+1
    J(c)=1/(2*length(H))*sum(-Y'*log(H')-(1-Y')*log(1-H'))+l/(2*length(H))*sum(theta.^2);
    if J(c-1)-J(c)<0
        break
    end
    diff=(J(c-1)-J(c))./J(c-1);
    if diff <.0001
        stop=true;
    end
   
end
figure();
plot(J);
end
