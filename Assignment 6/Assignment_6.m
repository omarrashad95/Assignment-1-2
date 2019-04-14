clear
close all
clc
%%%%%%%%%%%%% reading data%%%%%%%%%%%%%%%%%%%%%%%%
ds = tabularTextDatastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T=read(ds);

%%%%%%%%%%%%%%%% normalizing data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x=T{:,4:21};
[m,n]=size(x);
Y=T{:,3}/max(T{:,3});
Data_Scaled=zeros(size(x));
for i=1:size(x,2)
    Data_Scaled(:,i)=x(:,i)/max(x(:,i));
end

%%%%%%%%%%%%%% calculating correlation and covariance matricies%%%%%%%%%%
Corr_x=corr(Data_Scaled);
x_cov=cov(Data_Scaled);

%%%%%%%%%% pricipal components analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[U S V]=svd(x_cov);
EigenValues=diag(S)';
k=1;
while(true)
    alpha=1-(sum(EigenValues(1:k))/sum(EigenValues(1:18)));
    if(alpha <= 0.001)
        break;
    end
    k=k+1;
end
Reduced_Data=U(:,1:k)'* Data_Scaled';
App_Data=Reduced_Data'*V(1:k,:);
Error=(1/17999).* sum((App_Data(:,1:k)'-Reduced_Data).^2);

%%%%%%% linear regresion on reduced data %%%%%%%%%%%%%%%%%%%%%%%%%
X=[ones(1,length(Reduced_Data));Reduced_Data];
theta1=zeros(1,k+1); %theta parameters
H1=theta1*X; %forming the hypothesis
[JJ1 theta1]=Linear_Regression(H1,Y,X,theta1);

%%%%%%%%% K-Means Clustering %%%%%%%%%%%%%%%%%%%%%%%%%%%

%cluster centroid initialization
MIN=[];
for K=1:10
    J=[];
    C_Optimal=[];
    miu_Optimal=[];
    miu_ci_Optimal=[];
    for i=1:100
        [C,miu,miu_ci]=K_Means(Reduced_Data,K);
        J=[J sum(sum((Reduced_Data-miu_ci).^2))/length(Reduced_Data)];
    end
    MIN(K)=min(J);
end
K=1:10;
figure();
plot(K,smooth(MIN));
title('K-Means Clustering on the Reduced Data');
xlabel('Number of Clusters');
ylabel('Distortion Function');
%--------------------------------------------------------------------------
MIN2=[];
Real_Data=Data_Scaled';
for K2=1:10
J2=[];
C2_Optimal=[];
miu2_Optimal=[];
miu2_ci_Optimal=[];
for i=1:100
    [C2,miu2,miu2_ci]=K_Means(Real_Data,K2);
    J2=[J2 sum(sum((Real_Data-miu2_ci)).^2)/length(Real_Data)];
end
MIN2(K2)=min(J2);
end
K2=1:10;
figure();
plot(K2,smooth(MIN2));
title('K-means Clustering on The Real Data');
xlabel('Number of Clusters');
ylabel('Distortion Function');

%%%%%%%%%%%%%%%%%%%%%% Anomally Detection %%%%%%%%%%%%%%%%%
Training=m*0.7;
TrainingSet=x(1:Training,:);
TestingSet=x(Training+1:end,:);
Mean=mean(TrainingSet);
Sigma=cov(TrainingSet);
P=mvnpdf(TestingSet,Mean,Sigma);
epsilon=10^-30;
AnomallyDetection=(P>=epsilon);