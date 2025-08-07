function l_lik=log_likelihood(paramest0,opt)
global lct lyt lht
global param
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=length(lct);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opt==1
    % untransformed parameters
    thetab=exp(paramest0(1));
    rho=exp(paramest0(2))/(1+exp(paramest0(2)));
    sigmae=exp(paramest0(3));
    rhoa=exp(paramest0(4))/(1+exp(paramest0(4)));
    sigmaa=exp(paramest0(5));
    %delta=exp(paramest0(8))/(1+exp(paramest0(8)));
    alpha=exp(paramest0(6))/(1+exp(paramest0(6)));
    gamma=exp(paramest0(7));
else
    %parameters
    thetab=paramest0(1);
    rho=paramest0(2);
    sigmae=paramest0(3);
    rhoa=paramest0(4);
    sigmaa=paramest0(5);
    %delta=paramest0(8);
    alpha=paramest0(6);
    gamma=paramest0(7);
end
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
    
% if control variables are yt ht: 
% MCT=MBT1;

% if control variables are ct ht:
 MCT=[MCT1; MBT1(2,:)];
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LCt=lct'-log(css); % LCt=lct';
LYt=lyt'-log(yss); % LYt=lyt'; 
Lht=lht'-log(hss); % Lht=lht';
lcontrol=[LCt; Lht]; 
     
lestado=zeros(3,1);Mlestado=[];
MSIGMA=[sigmae^2 0;
        0        sigmaa^2];
bigxi=MI*MSIGMA*MI';             
bigP1=inv(eye(9)-kron(MT,MT))*bigxi(:);
bigPt=reshape(bigP1,3,3);

l_lik=(2*T/2)*log(2*pi);
for t=1:T
    %t
    ut=lcontrol(t,:)'-MCT*lestado;
    omegt=MCT*bigPt*MCT';
    omeginvt=inv(omegt);
    l_lik=l_lik+(1/2)*log(det(omegt))+(1/2)*ut'*omeginvt*ut;
    bigkt=MT*bigPt*MCT'*omeginvt;
    lestado=MT*lestado+bigkt*ut;Mlestado=[Mlestado lestado];
    bigPt=(MT-bigkt*MCT)*bigPt*(MT'-MCT'*bigkt')+MI*MSIGMA*MI';
    %MPHI1*bigPt*MPHI1'-bigkt*MFC1*bigPt*MPHI1'+MPHI3*MSIGMA*MPHI3';
    %(MPHI1-bigkt*MFC1)*bigPt*(MPHI1'-MFC1'*bigkt')+MPHI3*MSIGMA*MPHI3';
end

