var Y I C L W R K A;
varexo e;

parameters sigma phi alpha beta delta rhoa;
sigma = 2.0727;
phi = 1.4561; 
alpha = 0.3152; 
beta = 0.9854; 
delta = 0.0245; 
rhoa = 0.9056;

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

//-Steady state calculation
steady;
check; 
model_diagnostics; 
model_info;

//-Shock simulations
shocks;
var e; 
stderr 0.0104; 
end;

stoch_simul(order=1, irf=20, hp_filter = 1600, contemporaneous_correlation) Y I C L W R K A;
