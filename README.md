# Real Business Cycles

Read more about calibration
- [Dynare's manual](https://archives.dynare.org/manual/Estimation.html)
- [Computational Methods for Economics](https://opensourceecon.github.io/CompMethods/struct_est/GMM.html)
- [Chad Fulton's notes on RBC calibration](https://www.chadfulton.com/topics/simple_rbc.html#calibration-maximum-likelihood-estimation)
- [Sims' note](https://sites.nd.edu/esims/courses/ph-d-macro-theory-ii/)
- ABCs of RBC, King and Rebelo (1999)

## Baseline model

Preferences

$$
u(C_t,L_t) = \frac{C_t^{1-\sigma}}{1-\sigma} - \frac{L_t^{1+\varphi}}{1+\varphi}
$$

and a Cobb-Douglas production function

$$
Y_t = A_t K_t^\alpha L_t^{1-\alpha}
$$

This is a very simple and standard RBC model.

## Dynare codes

See the `rbc_log.mod` file.

The last line produces the IRF.

![](https://raw.githubusercontent.com/thanhqtran/gso-macro-monitor/main/generated_gif/irf.png)

To extract the elements for the state-space representation, please read [Sims' note](https://sites.nd.edu/esims/files/2023/05/using_dynare_sp17.pdf#page=8.77))

```matlab
% extract the parameters for state-space representation
% variable order: Y I C L W R K A
% state vars: 7, 8
p_Y = 1;
p_I = 2;
p_C = 3;
p_L = 4;
p_W = 5;
p_R = 6;
p_K = 7;
p_A = 8;

% create matrices for the state-space representation
% S(t) = A*S(t-1) + B*e(t)
% X(t) = C*S(t-1) + D*e(t)

A = [   oo_.dr.ghx(oo_.dr.inv_order_var(p_K),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_A),:)
    ];
B = [   oo_.dr.ghu(oo_.dr.inv_order_var(p_K),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_A),:)
    ];
C = [   oo_.dr.ghx(oo_.dr.inv_order_var(p_Y),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_I),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_C),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_R),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_W),:);
        oo_.dr.ghx(oo_.dr.inv_order_var(p_L),:);
    ];
D = [   oo_.dr.ghu(oo_.dr.inv_order_var(p_Y),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_I),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_C),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_R),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_W),:);
        oo_.dr.ghu(oo_.dr.inv_order_var(p_L),:);
    ];
```
The `console.m` file contains the code to replicate the IRFs and simulations.

The following shows the results of a simulated data based on a series of random shocks.

![](https://raw.githubusercontent.com/thanhqtran/gso-macro-monitor/main/generated_gif/simulated_random.png)

## Model performance

### Data retrieval

Run `usdata.py`. The business cycle moments are stored in the Dataframe `cycle`. 


![](https://raw.githubusercontent.com/thanhqtran/gso-macro-monitor/main/generated_gif/data.png)


### Structural parameter estimation

This procedure can be done in Dynare by adding the following code block to the end of the `.mod` file. Make sure that the `rbc_data.csv` is stored at the same location as the `.mod` file.

```matlab
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
```
The new model file `rbc_log_recalibrated.mod` has its parameters updated.

![](https://raw.githubusercontent.com/thanhqtran/gso-macro-monitor/main/generated_gif/bayesian.png)


| Parameter | Prior Mean | Posterior Mean | 90% HPD Interval | Prior | Std. Dev. |
| --------- | ---------- | -------------- | ---------------- | ----- | --------- |
| sigma     | 2.000      | 2.0727         | [1.2735, 2.8675] | norm  | 0.5000    |
| phi       | 1.500      | 1.4561         | [0.9421, 1.9585] | norm  | 0.3000    |
| alpha     | 0.330      | 0.3152         | [0.2291, 0.3925] | beta  | 0.0500    |
| beta      | 0.985      | 0.9854         | [0.9775, 0.9932] | beta  | 0.0050    |
| delta     | 0.025      | 0.0245         | [0.0163, 0.0328] | beta  | 0.0050    |
| rhoa      | 0.950      | 0.9056         | [0.8591, 0.9498] | beta  | 0.0200    |

One can re-run the model with the Bayesian updated parameters.

The table below shows the business cycle statistics of the model.

| Variable | Std. Dev | Rel. Std. Dev | First-order AR | Contemp. Corr w/ Y |
| -------- | -------- | ----------------- | -------------- | ------------------------- |
| Y        | 0.0157   | 1.0000            | 0.6993         | 1.0000                    |
| I        | 0.0654   | 4.1663            | 0.6912         | 0.9939                    |
| C        | 0.0041   | 0.2579            | 0.7867         | 0.9005                    |
| L        | 0.0036   | 0.2314            | 0.7035         | 0.9126                    |
| W        | 0.0125   | 0.7945            | 0.7149         | 0.9929                    |
| R        | 0.0164   | 1.0438            | 0.6949         | 0.9446                    |
| K        | 0.0054   | 0.3428            | 0.9549         | 0.3359                    |
| A        | 0.0133   | 0.8521            | 0.6948         | 0.9987                    |    

These are the statistics from the data.

| Variable | Std. Dev    | Rel. Std. Dev |
| -------- | --------- | --------- |
| Y        | 0.0190  | 1         |
| I        | 0.0611  | 3.2222   |
| C        | 0.0135   | 0.7152   |
| L        | 0.0093 | 0.4881   |
| K        | 0.0351  | 1.8490     |
| eA       | 0.0082 | 0.4315   |

Based on how closely the model performs compared to the real data, we can judge the goodness of fit of the model.

Although this simple RBC model does not capture the economy described in the data very well, it does provide a useful starting point.

### Eyeball Simulation

See the `console.m`

![](https://raw.githubusercontent.com/thanhqtran/gso-macro-monitor/main/generated_gif/simulated.png)


