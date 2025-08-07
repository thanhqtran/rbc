%Edited by Jesus Ruiz (jruizand@ucm.es)
clear;
clc;
close all;
%
global lct lyt lht
global param 
% loading data series
% data structure: y,c,h,year,quarter
data = readmatrix("ychvn.csv");
lyt=log(data(:,1)');
lct=log(data(:,2)');
lht=log(data(:,3)');
year_data = data(:,4);
quarter_data = data(:,5);
trend=1:length(lct);
T=length(lyt);
% % Create date labels for plots
date_labels = cell(T, 1);
for i = 1:T
    date_labels{i} = sprintf('%d:Q%d', year_data(i), quarter_data(i));
end
% trend estimation
trend_vec = (1:T)'; % Create a column vector for the time trend
X = [ones(T, 1), trend_vec]; % Create the design matrix [intercept, trend]
% Estimate trend for log output and detrend
coeffs_y = X \ lyt'; % OLS regression: coeffs = inv(X'*X)*X'*y
slope_y = coeffs_y(2);
clyt = lyt' - slope_y * trend';
% Estimate trend for log consumption and detrend
coeffs_c = X \ lct';
slope_c = coeffs_c(2);
clct=lct' - slope_c*trend'; 
% detrended series
lct=clct';
lyt=clyt';
% Demean the data after detrending. The model explains
% fluctuations around a zero mean. This is the empirical counterpart
% to the log-deviations from the steady state.
lyt = lct - mean(lct);
lct = lyt - mean(lyt);
lht = lht - mean(lht);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% calibrated parameter
% annual interest rate
annual_r = 9.113996795/100;     %IFS: FILR_PA
annual_dep = 0.057428208;       %pwt
r_target = (1+annual_r)^(1/4)-1;
delta = 1-(1-annual_dep)^(1/4);
beta = 1/(r_target+1-delta);
% untransformed parameters
thetab=1.6;       
alpha=0.26;    
rho=0.8;      
sigmae=0.007;  
rhoa=0.8;     
sigmaa=0.007;  
gamma=0.0045;  
%%%%%%%%%%%%%%%%%%%%%%%%%%
% transformed parameters
thetabtr=log(thetab);
betatr=log(beta/(1-beta));
deltatr=log(delta/(1-delta));
alphatr=log(alpha/(1-alpha));
rhotr=log(rho/(1-rho));
sigmaetr=log(sigmae);
rhoatr=log(rhoa/(1-rhoa));
sigmaatr=log(sigmaa);
gammatr=log(gamma);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parameters to be estimated
paramest0=[thetabtr rhotr sigmaetr rhoatr sigmaatr alphatr gammatr];
% exogenously given parameters
param=[beta delta];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Estimation  
options=optimset('Display','iter','MaxFunEvals',10000,...
    'MaxIter',10000,'TolFun',0.0001);
paramest1=fminunc('log_likelihood',paramest0,options,1);  %fminsearch('log_likelihood',paramest0,options,1);
thetab=exp(paramest1(1));
rho=exp(paramest1(2))/(1+exp(paramest1(2)));
sigmae=exp(paramest1(3));
rhoa=exp(paramest1(4))/(1+exp(paramest1(4)));
sigmaa=exp(paramest1(5));
%delta=exp(paramest1(8))/(1+exp(paramest1(8)));
alpha=exp(paramest1(6))/(1+exp(paramest1(6)));
gamma=exp(paramest1(7));
paramstar=[thetab rho sigmae rhoa sigmaa alpha gamma];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% covariance matrix
pstar=paramstar';
n=length(pstar);    
H0=zeros(n,n);
Vepsi=ones(length(pstar),1).*1e-06;
Mepsi=diag(Vepsi);
for j=1:length(pstar)
    Mepsi(j,j)=max(pstar(j)*1e-04,1e-06);
end
    auxi=Mepsi;  
    for i=1:n
        for j=1:n
            H0(i,j)=(feval('log_likelihood',pstar+auxi(:,i)+auxi(:,j),0)-...
                        feval('log_likelihood',pstar+auxi(:,i),0)-...
                        feval('log_likelihood',pstar+auxi(:,j),0)+...
                        feval('log_likelihood',pstar,0))/(auxi(j,j)*auxi(i,i));
        end
    end   
informd=inv(H0);
sd=abs(sqrt(diag(informd)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Table 1. Maximum Likelihood Estimation');
coefficient=pstar;
Std_Error=sd;
t_statistic=pstar./Std_Error;
p_value=(1-normcdf(abs(t_statistic),0,1)).*2;
names_var={'productivity shock mean (theta bar)'; 'productivity persistence (rho)';
           'productivity innovation std (sigma_e)'; 'preference shock persistence (rho_a)';
           'innovation of preference shock std (sigma_a)';
           'output elasticity of capital (alpha)'; 'utility function parameter (gamma)'};
TABLA=table(coefficient,Std_Error,t_statistic,p_value,'RowNames',names_var)
disp('    Calibrated parameter')   
disp(['    Discount factor (beta):          '  num2str(beta)])
disp(['    Depreciation rate:               '  num2str(delta)])
disp(['___________________________________________________________________'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Ax,Bx,Cx,V]=state_space_matrices(paramstar);
[V_variance, Mdecomp_v]=decomp_var(Ax,Bx,Cx,V);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
quarters=[1;4;8;12;20;40;inf];
names_var2={'1-step forecast';'4-step forecast';'8-step forecast';
    '12-step forecast';'20-step forecast';'40-step forecast';
    'inconditional variance (inf-step forecast)'};
%
supply_shock=Mdecomp_v(1:7,1);
demand_shock=Mdecomp_v(1:7,2);
disp('====================================================================')
disp('Table 2.Forecast error variance decompositions. Consumption')
TABLE=table(quarters,supply_shock,demand_shock,'RowNames',names_var2)
%
supply_shock=Mdecomp_v(8:14,1);
demand_shock=Mdecomp_v(8:14,2);
disp('====================================================================')
disp('Table 3.Forecast error variance decompositions. Output')
TABLE=table(quarters,supply_shock,demand_shock,'RowNames',names_var2)
%
supply_shock=Mdecomp_v(15:21,1);
demand_shock=Mdecomp_v(15:21,2);
disp('====================================================================')
disp('Table 4.Forecast error variance decompositions. Hours Worked')
TABLE=table(quarters,supply_shock,demand_shock,'RowNames',names_var2)
%
supply_shock=Mdecomp_v(22:28,1);
demand_shock=Mdecomp_v(22:28,2);
disp('====================================================================')
disp('Table 5.Forecast error variance decompositions. Investment')
TABLE=table(quarters,supply_shock,demand_shock,'RowNames',names_var2)
%
supply_shock=Mdecomp_v(29:35,1);
demand_shock=Mdecomp_v(29:35,2);
disp('====================================================================')
disp('Table 6.Forecast error variance decompositions. Wages')
TABLE=table(quarters,supply_shock,demand_shock,'RowNames',names_var2)
%
supply_shock=Mdecomp_v(36:42,1);
demand_shock=Mdecomp_v(36:42,2);
disp('====================================================================')
disp('Table 7.Forecast error variance decompositions. Interest rate')
TABLE=table(quarters,supply_shock,demand_shock,'RowNames',names_var2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%impulse-responses
    % supply shock
        innovs=zeros(100,2);
        innovs(10,1)=sigmae;
        imps=irf(paramstar,innovs);
        imps=imps*100;
    % demand shock
        innovd=zeros(100,2);
        innovd(10,2)=sigmaa;
        impd=irf(paramstar,innovd);
        impd=impd*100;

    %Graphs
    figure;
    hold on
    subplot(4,2,1)
    H1=plot([imps(:,1) impd(:,1)]);
    title('capital (%)');
    xlabel('periods');
    set(H1(1),'Color','k','LineStyle','--');
    set(H1(2),'Color','r','LineStyle','-');
    
    subplot(4,2,2)
    H2=plot([imps(:,2) impd(:,2)]);
    title('productivity shock (%)');
    xlabel('periods');
    set(H2(1),'Color','k','LineStyle','--');
    set(H2(2),'Color','r','LineStyle','-');
    
    subplot(4,2,3)
    H3=plot([imps(:,3) impd(:,3)]);
    title('preferences shock (%)');
    xlabel('periods');
    set(H3(1),'Color','k','LineStyle','--');
    set(H3(2),'Color','r','LineStyle','-');
    
    subplot(4,2,4)
    H4=plot([imps(:,4) impd(:,4)]);
    title('consumption (%)');
    xlabel('periods');
    set(H4(1),'Color','k','LineStyle','--');
    set(H4(2),'Color','r','LineStyle','-');
    
    subplot(4,2,5)
    H5=plot([imps(:,5) impd(:,5)]);
    title('output (%)');
    xlabel('periods');
    set(H5(1),'Color','k','LineStyle','--');
    set(H5(2),'Color','r','LineStyle','-');
    
    subplot(4,2,6)
    H6=plot([imps(:,6) impd(:,6)]);
    title('hours worked (%)');
    xlabel('periods');
    set(H6(1),'Color','k','LineStyle','--');
    set(H6(2),'Color','r','LineStyle','-');
    
    subplot(4,2,7)
    H7=plot([imps(:,7) impd(:,7)]);
    title('investment (%)');
    xlabel('periods');
    set(H7(1),'Color','k','LineStyle','--');
    set(H7(2),'Color','r','LineStyle','-');
    
    subplot(4,2,8)
    H8=plot([imps(:,8) impd(:,8)]);
    title('interest rate (%)');
    xlabel('periods');
    set(H8(1),'Color','k','LineStyle','--');
    set(H8(2),'Color','r','LineStyle','-');
    
    legend('supply shock','demand shock');
    hold off
