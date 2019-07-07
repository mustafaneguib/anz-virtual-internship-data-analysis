// Authors: 
//   Neil Marchant, October 2018
//   Andrew Bennett, August 2015
//
// Stan model for the nuclear power plant example
// from the University of Melbourne COMP90051 lectures

data {
  int<lower=1> N; // number of observations
  // for each of these, 0 means false, 1 means true
  int<lower=0, upper=1> HT[N];  // whether core temperature high
  int<lower=0, upper=1> FG[N];  // whether gauge is faulty
  int<lower=0, upper=1> FA[N];  // whether alarm is faulty
  int<lower=0, upper=1> HG[N];  // whether gauge reads high
  int<lower=0, upper=1> AS[N];  // whether alarm sounds

  real<lower=0> ALPHA; // alpha to use for beta distributions
  real<lower=0> BETA; // beta to use for beta distributions
}

parameters {
  //probability table for each variable
  real<lower=0, upper=1> p_ht;
  real<lower=0, upper=1> p_fg;  
  real<lower=0, upper=1> p_fa;
  real<lower=0, upper=1> p_hg[2,2];
  real<lower=0, upper=1> p_as[2,2];
}

model {
  // priors for probability table parameters
  p_ht ~ beta(ALPHA, BETA);
  p_fg ~ beta(ALPHA, BETA);
  p_fa ~ beta(ALPHA, BETA);
  for (i in 1:2) {
    for (j in 1:2) {
      p_hg[i, j] ~ beta(ALPHA, BETA); 
      p_as[i, j] ~ beta(ALPHA, BETA);
    }
  }
  
  // likelihood of observing data
  for (n in 1:N) {
    HT[n] ~ bernoulli(p_ht);
    FG[n] ~ bernoulli(p_fg);
    FA[n] ~ bernoulli(p_fa);
    // +1 in indexing below because Stan arrays count from 1
    HG[n] ~ bernoulli(p_hg[HT[n]+1, FG[n]+1]);
    AS[n] ~ bernoulli(p_as[FA[n]+1, HG[n]+1]);
  }
}
