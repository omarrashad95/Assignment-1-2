clc
clear
close all
ds = tabularTextDatastore('house_data_complete.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T=read(ds);
Data=[];
for i=3:size(T,2)
    if(i==8||i==17)
        temp=str2num(char(T{:,i}));
        Data=[Data temp];
    else
        temp=T{:,i};
        Data=[Data temp];
    end
end
Data_Scaled=zeros(size(Data));
for i=1:size(Data,2)
    Data_Scaled(:,i)=Data(:,i)/max(Data(:,i));
end
Train_Data=Data_Scaled(1:15130,2:end); %70 percent of complete data
Train_Outputs=Data_Scaled(1:15130,1);
Test_Data=Data_Scaled(15131:end,2:end);% 30 percent of complete data
Test_Outputs=Data_Scaled(15131:end,1); 

%features that have the most effect on the price according to covariance and data visualization.

x1=Train_Data(:,3);
x2=Train_Data(:,4);
x3=Train_Data(:,6);
x4=Train_Data(:,10);
x5=Train_Data(:,11);
x6=(Train_Data(:,12));
x7=(Train_Data(:,16));
x8=(Train_Data(:,18));


%---------------------------------1st hypothesis(Linear)

X1=[ones(1,length(Train_Data));exp(x4)';exp(-x6)';exp(x7)']; %set of all selected features of all training data
theta1=zeros(1,4); %theta parameters
H1=theta1*X1; %forming the hypothesis
[JJ1 theta1]=Linear_Regression(H1,Train_Outputs,X1,theta1);
x4_T1=Test_Data(:,10);
x6_T1=Test_Data(:,12);
x7_T1=Test_Data(:,16);
X_test1=[ones(1,length(Test_Data));exp(x4_T1)';exp(-x6_T1)';exp(x7_T1)'];
H_T1=theta1*X_test1; 
J_test1=(1/(2*length(H_T1)))*sum((H_T1-Test_Outputs').^2);

%-----------------------------------------------------2nd hypothesis(polynomial)

X2=[ones(1,length(Train_Data));sqrt(x1)';sqrt(x8)']; %set of all selected features of all training data
theta2=zeros(1,3); %theta parameters
H2=theta2*X2; %forming the hypothesis
[JJ2 theta2]=Linear_Regression(H2,Train_Outputs,X2,theta2);
x1_T2=Test_Data(:,3);
x8_T2=Test_Data(:,18);
X_test2=[ones(1,length(Test_Data));sqrt(x1_T2)';sqrt(x8_T2)'];
H_T2=theta2*X_test2; 
J_test2=(1/(2*length(H_T2)))*sum((H_T2-Test_Outputs').^2); 

%-------------------------------------------------------3rd hypothesis

X3=[ones(1,length(Train_Data));(x2)';(x3.^2)';(x5)']; %set of all selected features of all training data
theta3=zeros(1,4); %theta parameters
H3=theta3*X3; %forming the hypothesis
[JJ3 theta3]=Linear_Regression(H3,Train_Outputs,X3,theta3);
x2_T3=Test_Data(:,4);
x3_T3=Test_Data(:,6);
x5_T3=Test_Data(:,11);
X_test3=[ones(1,length(Test_Data));x2_T3';(x3_T3.^2)';x5_T3'];
H_T3=theta3*X_test3; 
J_test3=(1/(2*length(H_T3)))*sum((H_T3-Test_Outputs').^2); 

%-------------------------------------------------------4th hypothesis

X4=[ones(1,length(Train_Data));sqrt(x1)';(x2)';(x3.^2)';exp(x4)';(x5)';exp(-x6)';exp(x7)';sqrt(x8)']; %set of all selected features of all training data
theta4=zeros(1,9); %theta parameters
H4=theta4*X4; %forming the hypothesis
[JJ4 theta4]=Linear_Regression(H4,Train_Outputs,X4,theta4);
x1_T4=Test_Data(:,3);
x2_T4=Test_Data(:,4);
x3_T4=Test_Data(:,6);
x4_T4=Test_Data(:,10);
x5_T4=Test_Data(:,11);
x6_T4=(Test_Data(:,12));
x7_T4=(Test_Data(:,16));
x8_T4=(Test_Data(:,18));
X_test4=[ones(1,length(Test_Data));sqrt(x1_T4)';(x2_T4)';(x3_T4.^2)';exp(x4_T4)';(x5_T4)';exp(-x6_T4)';exp(x7_T4)';sqrt(x8_T4)'];
H_T4=theta4*X_test4; 
J_test4=(1/(2*length(H_T4)))*sum((H_T4-Test_Outputs').^2); 
