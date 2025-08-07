function imp=irf(paramest0,innov)
global param
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=length(innov);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parameters
thetab=paramest0(1);
rho=paramest0(2);
sigmae=paramest0(3);
rhoa=paramest0(4);
sigmaa=paramest0(5);
%delta=paramest0(8);
alpha=paramest0(6);
gamma=paramest0(7);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% other parameters
beta=param(1);
delta=param(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  steady state 
thetass=thetab;
ass=1;
css=((1-alpha)/gamma)*thetass*...
    (thetass*alpha/((1/beta)-(1-delta)))^(alpha/(1-alpha));
kss=alpha*css/((1/beta)-(1-delta)-alpha*delta);
hss=kss*(((1/beta)-(1-delta))/(thetass*alpha))^(1/(1-alpha));
yss=thetass*(kss^alpha)*(hss^(1-alpha));
iss=delta*kss;
Rss=alpha*thetass*(kss/hss)^(alpha-1)+1-delta;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Computing matrices A, B, C, D, F, G, H, J, P, K, L

MA=[1                     0               -(1-alpha);
    (1/beta)-(1-delta)    -alpha*delta    0;
    1                     0               -1];

MB=[alpha      0;
    0          (1/beta)-(1-delta)-alpha*delta;
    0          1];

MC=[1  0;
    0  0;
    0  0];

MD=[1                     0;
    (1/beta)-(1-delta)    1/beta];

MF=[0                       0    0;
    -((1/beta)-(1-delta))   0    0];

MG=[1-delta     0;
    0           1/beta];

MH=[0  delta  0;
    0  0      0];

MJ=[0          0;
    0          -(1/beta)*(1-rhoa)];

MK=inv(MD+MF*inv(MA)*MB)*(MG+MH*inv(MA)*MB);

MP=[rho 0;
    0   rhoa];

ML=inv(MD+MF*inv(MA)*MB)*(MJ+MH*inv(MA)*MC-MF*inv(MA)*MC*MP);

%  Computing matrices m ,  Q

  [kvec,keig]=eig(MK);
  
  keig2=diag(keig);

  [keig3,kord]=sort(abs(keig2));

  if keig3(1)>1
     'error - no solution'
  end

  if keig3(2)<1
     'error - multiple solutions'
  end

  keig4=keig2(kord);

  Mlambda=diag(keig4);

  MM=kvec(:,kord);

  Mm=inv(MM);

  MQ=Mm*ML;
  
%  Computing trajectories for capital stock, technology shock and consumption 
  
G1=(MQ(1,1)-((rho-Mlambda(1,1))/(rho-Mlambda(2,2)))*(MQ(2,1)*Mm(1,2)/Mm(2,2)))/...
    (Mm(1,1)-Mm(1,2)*Mm(2,1)/Mm(2,2));
G2=(MQ(1,2)-((rhoa-Mlambda(1,1))/(rhoa-Mlambda(2,2)))*(MQ(2,2)*Mm(1,2)/Mm(2,2)))/...
    (Mm(1,1)-Mm(1,2)*Mm(2,1)/Mm(2,2));

MT=[Mlambda(1,1)  G1   G2;
   0              rho  0
   0              0    rhoa];    %  State transition matrix

MI=[0 0;
    1 0;
    0 1];     %   Matrix of effects of the innovation on the states

%matirx of control equations
MCT1=[-Mm(2,1)/Mm(2,2)    (MQ(2,1)/Mm(2,2))/(rho-Mlambda(2,2))   (MQ(2,2)/Mm(2,2))/(rhoa-Mlambda(2,2))];     %   Control matrix (control: ct)

    MA1=[1  -(1-alpha);
         1  -1];
    
    MB1=[alpha               1                                      0;
         -Mm(2,1)/Mm(2,2)    (MQ(2,1)/Mm(2,2))/(rho-Mlambda(2,2))   (MQ(2,2)/Mm(2,2))/(rhoa-Mlambda(2,2))];
    
    MBT1=inv(MA1)*MB1;  

MCT=[MCT1;
     MBT1];
 
ESTAD=zeros(3,length(innov)+1);

for t=1:length(innov)
    ESTAD(:,t+1)=MT*ESTAD(:,t)+MI*innov(t,:)';
end
CONTROL=MCT*ESTAD;

kt=ESTAD(1,:)';
thetat=ESTAD(2,:)';
at=ESTAD(3,:)';

ct=CONTROL(1,:)';
yt=CONTROL(2,:)';
ht=CONTROL(3,:)';
it=(1/(alpha*delta)).*(((1/beta)-1+delta).*yt-...
    ((1/beta)-1+delta-alpha*delta).*ct);
Rt=((Rss+delta-1)/Rss).*(thetat-(1-alpha).*(kt-ht));
imp=[kt thetat at ct yt ht it Rt];
