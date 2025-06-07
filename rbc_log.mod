// RBC model - Chapter 2 (UNDERSTANDING DSGE MODELS) 
var Y I C L W R K A;
varexo e;

parameters sigma phi alpha beta delta rhoa;
sigma = 2;
phi = 1.5; 
alpha = 0.33; 
beta = 0.985; 
delta = 0.025; 
rhoa = 0.95;

model(linear);
#Rss = (1/beta)-(1-delta);
#Wss = (1-alpha)*((alpha/Rss)^(alpha/(1-alpha))); 
#Yss = ((Rss/(Rss-delta*alpha))^(sigma/(sigma+phi))) *(((1-alpha)^(-phi))*((Wss)^(1+phi)))^(1/(sigma+phi));
#Kss = alpha*(Yss/Rss);
#Iss = delta*Kss;
#Css = Yss - Iss;
#Lss = (1-alpha)*(Yss/Wss);

//1-Labor supply
sigma*C + phi*L = W;
//2-Euler equation
(sigma/beta)*(C(+1)-C)=Rss*R(+1);
//3-Law of motion of capital
K = (1-delta)*K(-1)+delta*I;
//4-Production function
Y = A + alpha*K(-1) + (1-alpha)*L;
//5-Demand for capital
R = Y - K(-1);
//6-Demand for labor
W = Y - L;
//7-Equilibrium condition
Yss*Y = Css*C + Iss*I;
//8-Productivity shock
A = rhoa*A(-1) + e;
end;

// ---------- Observed variables ----------
varobs Y;

// ---------- Priors for Bayesian estimation ----------
estimated_params;
sigma, normal_pdf, 2, 0.5;
phi, normal_pdf, 1.5, 0.3;
alpha, beta_pdf, 0.33, 0.05;
beta, beta_pdf, 0.985, 0.005;
delta, beta_pdf, 0.025, 0.005;
rhoa, beta_pdf, 0.95, 0.02;
stderr e, inv_gamma_pdf, 0.01, 2;
end;

// ---------- Use calibration as initial values ----------
estimated_params_init(use_calibration);
end;

// ---------- Estimation command ----------
estimation(datafile='rbc_data.csv', first_obs=1, mh_replic=20000, mh_nblocks=2, mh_jscale=0.2, mode_compute=6);
