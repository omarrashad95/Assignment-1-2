clc
clear
close all
load('heartDD.mat');

x1=b{:,1};
x2=b{:,2};
x3=b{:,3};
x31=zeros(size(x3,1),1);
x32=x31;
x33=x31;
x31(x3==1)=1;
x32(x3==2)=1;
x33(x3==3)=1;
x4=b{:,4};
x5=b{:,5};
x6=b{:,6};
x7=b{:,7};
x8=b{:,8};
x9=b{:,9};
x10=b{:,10};
x11=b{:,11};
x11_1=zeros(size(x11,1),1);
x11_2=x11_1;
x11_1(x11==1)=1;
x11_2(x11==2)=1;
x12=b{:,12};
x12_1=zeros(size(x12,1),1);
x12_2=x12_1;
x12_3=x12_1;
x12_4=x12_1;
x12_1(x12==1)=1;
x12_2(x12==2)=1;
x12_3(x12==3)=1;
x12_4(x12==4)=1;
x13=b{:,13};
x13_1=zeros(size(x13,1),1);
x13_2=x13_1;
x13_3=x13_1;
x13_1(x13==1)=1;
x13_2(x13==2)=1;
x13_3(x13==3)=1;
X_all=[x1 x2 x31 x32 x33 x4 x5 x6 x7 x8 x9 x10 x11_1 x11_2 x12_1 x12_2 x12_3 x12_4 x13_1 x13_2 x13_3];
Data_Scaled=zeros(size(X_all));
for i=1:size(X_all,2)
    Data_Scaled(:,i)=X_all(:,i)/max(X_all(:,i));
end
Train_Data=Data_Scaled(1:175,:); %70 percent of complete data
Train_Outputs=b{1:175,end};
Test_Data=Data_Scaled(176:end,:);% 30 percent of complete data
Test_Outputs=b{176:end,end}; 
l=0.004;
% %---------------------------------1st hypothesis(Linear)

X1=[ones(1,length(Train_Data));Train_Data']; %set of all selected features of all training data
theta1=zeros(1,size(X1,1)); %theta parameters
H1=1./(1+exp(-(theta1*X1))); %forming the hypothesis
[JJ1 theta1]=Logistic_Regression(H1,Train_Outputs,X1,theta1);
X_test1=[ones(1,length(Test_Data)); Test_Data'];
H_T1=1./(1+exp(-(theta1*X_test1))); 
J_test1=1/(2*length(H_T1))*sum(-Test_Outputs'*log(H_T1')-(1-Test_Outputs')*log(1-H_T1'))+l/(2*length(H_T1))*sum(theta1(2:length(theta1)).^2);

%-----------------------------------------------------2nd hypothesis(polynomial)

X2=[ones(1,length(Train_Data));Train_Data';(Train_Data.^2)']; %set of all selected features of all training data
theta2=zeros(1,size(X2,1)); %theta parameters
H2=1./(1+exp(-(theta2*X2))); %forming the hypothesis
[JJ2 theta2]=Logistic_Regression(H2,Train_Outputs,X2,theta2);
X_test2=[ones(1,length(Test_Data)); Test_Data' ;(Test_Data.^2)'];
H_T2=1./(1+exp(-(theta2*X_test2))); 
J_test2=1/(2*length(H_T2))*sum(-Test_Outputs'*log(H_T2')-(1-Test_Outputs')*log(1-H_T2'))+l/(2*length(H_T2))*sum(theta2(2:length(theta2)).^2);

% %-------------------------------------------------------3rd hypothesis
X3=[ones(1,length(Train_Data));Train_Data';exp(Train_Data)']; %set of all selected features of all training data
theta3=zeros(1,size(X3,1)); %theta parameters
H3=1./(1+exp(-(theta3*X3))); %forming the hypothesis
[JJ3 theta3]=Logistic_Regression(H3,Train_Outputs,X3,theta3);
X_test3=[ones(1,length(Test_Data)); Test_Data' ;exp(Test_Data)'];
H_T3=1./(1+exp(-(theta3*X_test3))); 
J_test3=1/(2*length(H_T3))*sum(-Test_Outputs'*log(H_T3')-(1-Test_Outputs')*log(1-H_T3'))+l/(2*length(H_T3))*sum(theta3(2:length(theta3)).^2);

% %-------------------------------------------------------4th hypothesis
X4=[ones(1,length(Train_Data));Train_Data';(Train_Data.^2)';exp(Train_Data)']; %set of all selected features of all training data
theta4=zeros(1,size(X4,1)); %theta parameters
H4=1./(1+exp(-(theta4*X4))); %forming the hypothesis
[JJ2 theta4]=Logistic_Regression(H4,Train_Outputs,X4,theta4);
X_test4=[ones(1,length(Test_Data)); Test_Data' ;(Test_Data.^2)';exp(Test_Data)'];
H_T4=1./(1+exp(-(theta4*X_test4))); 
J_test4=1/(2*length(H_T4))*sum(-Test_Outputs'*log(H_T4')-(1-Test_Outputs')*log(1-H_T4'))+l/(2*length(H_T4))*sum(theta4(2:length(theta4)).^2);
