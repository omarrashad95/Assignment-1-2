function [J,theta]=Linear_Regression(H,Y,X,theta)

alpha=0.001; %learning rate
c=1;
J(c)=(1/(2*length(H)))*sum((H-Y').^2);%formulation of the MSE
stop=false;
%gradient descent
while stop==false
    for k=1:length(theta)
        theta(k)=theta(k)-(alpha/length(H))*((H-Y')*X(k,:)');
    end
    H=theta*X;
    c=c+1
    J(c)=(1/(2*length(H)))*sum((H-Y').^2);
    if J(c-1)-J(c)<0
        break
    end
    diff=(J(c-1)-J(c))./J(c-1);
    if diff <.001
        stop=true;
    end
   
end
figure();
plot(J);
title('Linear Regression on Reduced Data');
xlabel('Number of Iteraions');
ylabel('Cost Function');
end
